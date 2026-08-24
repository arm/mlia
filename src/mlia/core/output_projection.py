# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Conservative origin-based projection of standardized-output breakdowns.

Every authoritative breakdown is identified by the entity ID that owns the
original metric. That ID is the accounting origin: copies preserve it and sums
union distinct origins. Graph overlap is not an accounting collision. Two
separate chain entities may measure different scheduled portions of the same
source operator and remain additive when their metrics explicitly use ``sum``.

The normalized entity DAG determines complete structural coverage, hierarchy
competition, and contested extras. Terminal leaves are coverage units, while
authoritative entity IDs remain the accounting identities that distinguish
separate measurements even when their structural coverage overlaps.

Declared direct authoritative parents form a complete nearest frontier. Every
parent must be attribution-compatible. Compatible parents are combined once,
the result is sealed with their full origin set, and more distant authoritative
ancestors are shadowed. Downstream code-stack and module entities therefore use
the complete resolved frontier rather than reconsidering partial raw parents.
A conflicted narrow frontier blocks partial projection, but another target may
re-evaluate the complete origins when it covers the blocked scope and accounts
for every required origin. Normal contested-extra rules still apply.

Derived entities expose provenance recipes rather than becoming new accounting
origins. Recipe discovery reaches a fixed point before any dynamic target is
resolved, and every recipe retains the authoritative IDs and arithmetic that
produced it. Final resolution therefore compares the complete provenance closure
at once instead of depending on traversal order or prematurely committed values.

One complete contributor is copied without interpreting its aggregation policy.
Combining multiple distinct origins requires corresponding numeric metrics with
compatible explicit ``sum`` aggregation. A complete no-extra derivation outranks
one with uncontested extras. Same-priority conflicting results suppress
projection conservatively.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace
from typing import Any, cast

import mlia.core.output_schema as schema
from mlia.core.entity_graph import (
    EntityGraph,
    EntityGraphDeclaration,
    EntityGraphValidationError,
    validate_entity_graph,
)
from mlia.core.output_validation import (
    SchemaValidationError,
    validate_result_entity_kind_relationships,
)

MetricIdentity = tuple[str, str, str]
BreakdownSignature = tuple[str, str | None, str, tuple[str, ...]]
EntitySet = frozenset[str]

_PRIORITY_EXACT = 2
_PRIORITY_NO_EXTRAS = 3
_PRIORITY_RESIDUAL_COPY = 4
_PRIORITY_WITH_EXTRAS = 5
_MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True)
class _FigureBreakdown:
    """One canonical breakdown without entity-specific identity."""

    qualifiers_key: str
    qualifiers: dict[str, Any]
    metrics: tuple[schema.Metric, ...]


FigureSet = tuple[_FigureBreakdown, ...]


@dataclass(frozen=True)
class _AccountedFigures:
    """Canonical figures and their authoritative origin entity IDs."""

    figures: FigureSet
    origins: EntitySet
    sealed: bool = False


@dataclass(frozen=True)
class _EligibleContributor:
    """Semantically equivalent target-compatible figure sources."""

    entity_ids: frozenset[str]
    accounted: _AccountedFigures
    covered_units: EntitySet
    extras: EntitySet


EligibleContributorKey = tuple[
    tuple[str, ...],
    tuple[str, tuple[str, ...], bool],
    tuple[str, ...],
    tuple[str, ...],
]


@dataclass(frozen=True)
class _Derivation:
    """One complete way to produce a target's canonical figure set."""

    priority: int
    accounted: _AccountedFigures
    extras: EntitySet = frozenset()


@dataclass(frozen=True)
class _Resolution:
    """Resolved equal figures, or a conflict at the highest available priority."""

    records: tuple[_AccountedFigures, ...] = ()
    conflicted: bool = False


@dataclass(frozen=True)
class _FrontierResolution:
    """Resolved declared authoritative frontier, or a blocking conflict."""

    record: _AccountedFigures | None = None
    conflicted: bool = False
    required_origins: EntitySet = frozenset()


@dataclass(frozen=True)
class _FrontierRequirements:
    """Authoritative origins a target must represent to lift narrower conflicts."""

    origins: EntitySet = frozenset()
    conflicted: bool = False


def _canonical_json(value: Any) -> str:
    """Return a deterministic representation for schema values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def _metric_identity(metric: schema.Metric) -> MetricIdentity:
    """Return the fields that identify corresponding metrics."""
    return metric.name, metric.unit, _canonical_json(metric.qualifiers)


def _canonical_metric_signature(metric: schema.Metric) -> str:
    """Return a deterministic representation of a complete metric."""
    return _canonical_json(metric.to_dict())


def _figure_set_key(figure_set: FigureSet) -> str:
    """Return a deterministic equality key for a canonical figure set."""
    return _canonical_json(
        [
            {
                "qualifiers": figure.qualifiers,
                "metrics": [metric.to_dict() for metric in figure.metrics],
            }
            for figure in figure_set
        ]
    )


def _accounted_key(
    record: _AccountedFigures,
) -> tuple[str, tuple[str, ...], bool]:
    """Return a deterministic key including accounting provenance."""
    return (
        _figure_set_key(record.figures),
        tuple(sorted(record.origins)),
        record.sealed,
    )


def _normalize_metrics(
    metrics: list[schema.Metric],
) -> tuple[schema.Metric, ...] | None:
    """Canonicalize metrics, rejecting conflicting duplicate identities."""
    metrics_by_identity: dict[MetricIdentity, schema.Metric] = {}
    for metric in metrics:
        identity = _metric_identity(metric)
        previous = metrics_by_identity.get(identity)
        if previous is None:
            metrics_by_identity[identity] = metric
        elif _canonical_metric_signature(previous) != _canonical_metric_signature(
            metric
        ):
            return None

    return tuple(
        copy.deepcopy(metrics_by_identity[identity])
        for identity in sorted(metrics_by_identity)
    )


def _extract_figure_set(
    breakdowns: list[schema.Breakdown],
) -> FigureSet | None:
    """Return canonical figures, rejecting ambiguous duplicate breakdowns."""
    figures_by_qualifiers: dict[str, _FigureBreakdown] = {}
    for breakdown in breakdowns:
        metrics = _normalize_metrics(breakdown.metrics)
        if not metrics:
            return None

        qualifiers_key = _canonical_json(breakdown.qualifiers)
        figure = _FigureBreakdown(
            qualifiers_key=qualifiers_key,
            qualifiers=copy.deepcopy(breakdown.qualifiers),
            metrics=metrics,
        )
        previous = figures_by_qualifiers.get(qualifiers_key)
        if previous is None:
            figures_by_qualifiers[qualifiers_key] = figure
        elif previous != figure:
            return None

    if not figures_by_qualifiers:
        return None

    return tuple(figures_by_qualifiers[key] for key in sorted(figures_by_qualifiers))


def _validated_graph(result: schema.Result) -> EntityGraph:
    """Build the shared normalized DAG, translating failures to schema errors."""
    declarations = [
        EntityGraphDeclaration(
            id=entity.id,
            parent_ids=tuple(entity.parent_ids),
            child_ids=tuple(entity.child_ids),
            source_index=index,
        )
        for index, entity in enumerate(result.entities)
    ]
    try:
        graph = validate_entity_graph(declarations)
    except EntityGraphValidationError as err:
        details = "\n  - ".join(issue.message() for issue in err.issues)
        raise SchemaValidationError(
            f"Entity graph validation failed:\n  - {details}"
        ) from err
    validate_result_entity_kind_relationships(result, graph)
    return graph


def _declared_frontier_kind_edges(
    entity_kinds: list[schema.EntityKind],
) -> frozenset[tuple[str, str]]:
    """Return explicitly declared parent-to-child kind relationships."""
    edges = {
        (entity_kind.id, child_kind)
        for entity_kind in entity_kinds
        for child_kind in entity_kind.child_kinds
    }
    edges.update(
        (parent_kind, entity_kind.id)
        for entity_kind in entity_kinds
        for parent_kind in entity_kind.parent_kinds
    )
    return frozenset(edges)


def _aggregate_samples(metrics: list[schema.Metric]) -> int | None:
    """Combine additive sample counts, or signal incompatibility."""
    samples = [metric.samples for metric in metrics]
    if all(sample is None for sample in samples):
        return None
    if not all(type(sample) is int for sample in samples):
        raise ValueError("Incompatible metric sample metadata.")
    integer_samples = cast(list[int], samples)
    if any(sample < 0 or sample > _MAX_SAFE_INTEGER for sample in integer_samples):
        raise ValueError("Metric samples exceed the interoperable integer range.")
    total = sum(integer_samples)
    if total > _MAX_SAFE_INTEGER:
        raise ValueError("Aggregated metric samples exceed the interoperable range.")
    return total


def _aggregate_metric_group(metrics: list[schema.Metric]) -> schema.Metric | None:
    """Sum one corresponding numeric metric group conservatively."""
    values = [metric.value for metric in metrics]
    if not all(type(value) in (int, float) for value in values):
        return None
    if any(
        metric.availability is not None or metric.reason is not None
        for metric in metrics
    ):
        return None

    aggregations = {metric.aggregation for metric in metrics}
    if aggregations != {schema.AggregationType.SUM}:
        return None

    try:
        samples = _aggregate_samples(metrics)
    except ValueError:
        return None

    total = sum(cast(float | int, value) for value in values)
    if isinstance(total, float) and not math.isfinite(total):
        return None

    first = metrics[0]
    return schema.Metric(
        name=first.name,
        value=total,
        unit=first.unit,
        aggregation=schema.AggregationType.SUM,
        samples=samples,
        qualifiers=copy.deepcopy(first.qualifiers),
    )


def _aggregate_figure_sets(figure_sets: list[FigureSet]) -> FigureSet | None:
    """Aggregate compatible corresponding SUM metrics shared by all figures."""
    breakdown_maps = [
        {figure.qualifiers_key: figure for figure in figure_set}
        for figure_set in figure_sets
    ]
    common_breakdown_keys = set(breakdown_maps[0])
    for breakdown_map in breakdown_maps[1:]:
        common_breakdown_keys.intersection_update(breakdown_map)

    aggregated_breakdowns: list[_FigureBreakdown] = []
    for qualifiers_key in sorted(common_breakdown_keys):
        figures = [breakdown_map[qualifiers_key] for breakdown_map in breakdown_maps]
        metric_maps = [
            {_metric_identity(metric): metric for metric in figure.metrics}
            for figure in figures
        ]
        common_metric_ids = set(metric_maps[0])
        for metric_map in metric_maps[1:]:
            common_metric_ids.intersection_update(metric_map)

        aggregated_metrics = tuple(
            metric
            for metric_id in sorted(common_metric_ids)
            if (
                metric := _aggregate_metric_group(
                    [metric_map[metric_id] for metric_map in metric_maps]
                )
            )
            is not None
        )
        if aggregated_metrics:
            aggregated_breakdowns.append(
                _FigureBreakdown(
                    qualifiers_key=qualifiers_key,
                    qualifiers=copy.deepcopy(figures[0].qualifiers),
                    metrics=aggregated_metrics,
                )
            )

    return tuple(aggregated_breakdowns) or None


def _breakdowns_by_entity(
    breakdowns: list[schema.Breakdown],
) -> dict[str, list[schema.Breakdown]]:
    """Group authoritative breakdowns by their owning entity."""
    grouped: dict[str, list[schema.Breakdown]] = {}
    for breakdown in breakdowns:
        grouped.setdefault(breakdown.entity_id, []).append(breakdown)
    return grouped


def _inclusive_descendants(graph: EntityGraph) -> dict[str, EntitySet]:
    """Return each entity and every descendant reachable from it."""
    descendants: dict[str, EntitySet] = {}
    for entity_id in reversed(graph.topological_order):
        descendants[entity_id] = frozenset(
            {entity_id}
            | {
                descendant_id
                for child_id in graph.children[entity_id]
                for descendant_id in descendants[child_id]
            }
        )
    return descendants


def _scope_units(
    entity_id: str, structural_leaf_sets: dict[str, EntitySet]
) -> EntitySet:
    """Return the complete terminal structural coverage of one graph node."""
    return structural_leaf_sets[entity_id]


def _units_related(
    left_id: str,
    right_id: str,
    inclusive_descendants: dict[str, EntitySet],
) -> bool:
    """Return whether two graph branches intersect through explicit entities."""
    return bool(
        inclusive_descendants[left_id].intersection(inclusive_descendants[right_id])
    )


def _origin_units(
    authoritative_entity_ids: set[str],
    structural_leaf_sets: dict[str, EntitySet],
) -> dict[str, EntitySet]:
    """Return complete structural coverage for every authoritative origin."""
    return {
        entity_id: _scope_units(entity_id, structural_leaf_sets)
        for entity_id in authoritative_entity_ids
    }


def _authoritative_sources(
    breakdowns: list[schema.Breakdown],
) -> tuple[dict[str, tuple[_AccountedFigures, ...]], set[str]]:
    """Build records whose accounting identity is the owning entity ID."""
    grouped = _breakdowns_by_entity(breakdowns)
    states: dict[str, tuple[_AccountedFigures, ...]] = {}
    invalid_origin_ids: set[str] = set()
    for entity_id in sorted(grouped):
        figure_set = _extract_figure_set(grouped[entity_id])
        if figure_set is None:
            invalid_origin_ids.add(entity_id)
            continue
        states[entity_id] = (_AccountedFigures(figure_set, frozenset({entity_id})),)
    return states, invalid_origin_ids


def _record_origin_units(
    record: _AccountedFigures,
    origin_units: dict[str, EntitySet],
) -> EntitySet:
    """Return every graph branch represented by a record's origins."""
    return frozenset(
        unit_id for origin_id in record.origins for unit_id in origin_units[origin_id]
    )


def _covered_target_units(
    record: _AccountedFigures,
    target_units: EntitySet,
    origin_units: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
) -> EntitySet:
    """Return target branches intersected by at least one record origin."""
    represented_units = _record_origin_units(record, origin_units)
    return frozenset(
        target_unit
        for target_unit in target_units
        if any(
            _units_related(target_unit, represented_unit, inclusive_descendants)
            for represented_unit in represented_units
        )
    )


def _record_extras(
    record: _AccountedFigures,
    target_units: EntitySet,
    origin_units: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
) -> EntitySet:
    """Return represented origin branches outside the target hierarchy."""
    return frozenset(
        represented_unit
        for represented_unit in _record_origin_units(record, origin_units)
        if not any(
            _units_related(target_unit, represented_unit, inclusive_descendants)
            for target_unit in target_units
        )
    )


def _extra_is_contested(
    target_id: str,
    extra_id: str,
    entities_by_id: dict[str, schema.Entity],
    inclusive_ancestors: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
) -> bool:
    """Return whether a target competitor intersects an extra origin branch."""
    target = entities_by_id[target_id]
    target_ancestors = inclusive_ancestors[target_id]
    return any(
        other_id != target_id
        and (
            other.kind == target.kind
            or bool(target_ancestors.intersection(inclusive_ancestors[other_id]))
        )
        and _units_related(other_id, extra_id, inclusive_descendants)
        for other_id, other in entities_by_id.items()
    )


def _required_target_units(
    target_id: str,
    structural_leaf_sets: dict[str, EntitySet],
    origin_units: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
    shadowed_origin_ids: EntitySet,
) -> EntitySet:
    """Return target leaves intersected by non-shadowed authoritative origins."""
    target_units = _scope_units(target_id, structural_leaf_sets)
    all_origin_units = frozenset(
        unit_id
        for origin_id, units in origin_units.items()
        if origin_id not in shadowed_origin_ids
        for unit_id in units
    )
    return frozenset(
        target_unit
        for target_unit in target_units
        if any(
            _units_related(target_unit, origin_unit, inclusive_descendants)
            for origin_unit in all_origin_units
        )
    )


def _aggregate_records(records: list[_AccountedFigures]) -> _AccountedFigures | None:
    """Combine records with distinct authoritative origins."""
    if not records:
        return None
    if len(records) == 1:
        return records[0]
    origins = frozenset(origin for record in records for origin in record.origins)
    if sum(len(record.origins) for record in records) != len(origins):
        return None
    figures = _aggregate_figure_sets([record.figures for record in records])
    if figures is None:
        return None
    return _AccountedFigures(
        figures,
        origins,
        sealed=all(record.sealed for record in records),
    )


def _shadowed_authoritative_origins(
    graph: EntityGraph,
    authoritative_entity_ids: set[str],
    frontier_kind_edges: frozenset[tuple[str, str]],
    entities_by_id: dict[str, schema.Entity],
) -> frozenset[str]:
    """Return aggregate origins superseded by nearer declared authoritative data."""
    return frozenset(
        parent_id
        for parent_id in authoritative_entity_ids
        for child_id in graph.children[parent_id]
        if child_id in authoritative_entity_ids
        and (
            entities_by_id[parent_id].kind,
            entities_by_id[child_id].kind,
        )
        in frontier_kind_edges
    )


def _direct_authoritative_parent_resolutions(
    graph: EntityGraph,
    states: dict[str, tuple[_AccountedFigures, ...]],
    authoritative_entity_ids: set[str],
    frontier_kind_edges: frozenset[tuple[str, str]],
    entities_by_id: dict[str, schema.Entity],
    structural_leaf_sets: dict[str, EntitySet],
    origin_units: dict[str, EntitySet],
    inclusive_ancestors: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
    shadowed_origin_ids: EntitySet,
) -> dict[str, _FrontierResolution]:
    """Resolve every declared direct authoritative parent as one complete frontier."""
    resolutions: dict[str, _FrontierResolution] = {}
    for target_id in sorted(entities_by_id):
        if target_id in authoritative_entity_ids:
            continue
        target_kind = entities_by_id[target_id].kind
        parent_ids = sorted(
            parent_id
            for parent_id in graph.parents[target_id]
            if parent_id in authoritative_entity_ids
            and parent_id not in shadowed_origin_ids
            and (entities_by_id[parent_id].kind, target_kind) in frontier_kind_edges
        )
        if not parent_ids:
            continue

        target_units = _scope_units(target_id, structural_leaf_sets)
        records: list[_AccountedFigures] = []
        conflicted = False
        for parent_id in parent_ids:
            parent_records = states.get(parent_id, ())
            if len(parent_records) != 1:
                conflicted = True
                break
            record = parent_records[0]
            if not _covered_target_units(
                record,
                target_units,
                origin_units,
                inclusive_descendants,
            ):
                conflicted = True
                break
            extras = _record_extras(
                record,
                target_units,
                origin_units,
                inclusive_descendants,
            )
            if any(
                _extra_is_contested(
                    target_id,
                    extra_id,
                    entities_by_id,
                    inclusive_ancestors,
                    inclusive_descendants,
                )
                for extra_id in extras
            ):
                conflicted = True
                break
            records.append(record)

        required_origins = frozenset(parent_ids)
        if conflicted:
            resolutions[target_id] = _FrontierResolution(
                conflicted=True,
                required_origins=required_origins,
            )
            continue
        aggregate_record = _aggregate_records(records)
        resolutions[target_id] = (
            _FrontierResolution(
                record=replace(aggregate_record, sealed=True),
                required_origins=required_origins,
            )
            if aggregate_record is not None
            else _FrontierResolution(
                conflicted=True,
                required_origins=required_origins,
            )
        )
    return resolutions


def _frontier_requirements(
    target_id: str,
    conflicted_frontiers: dict[str, EntitySet],
    structural_leaf_sets: dict[str, EntitySet],
) -> _FrontierRequirements:
    """Return complete origins that another target must resolve together."""
    target_units = _scope_units(target_id, structural_leaf_sets)
    required_origins: set[str] = set()
    for blocked_id, origin_ids in conflicted_frontiers.items():
        blocked_units = structural_leaf_sets[blocked_id]
        if not target_units.intersection(blocked_units):
            continue
        if not blocked_units.issubset(target_units):
            return _FrontierRequirements(conflicted=True)
        if blocked_units == target_units and len(origin_ids) > 1:
            return _FrontierRequirements(conflicted=True)
        required_origins.update(origin_ids)
    return _FrontierRequirements(origins=frozenset(required_origins))


def _provenance_population(
    states: dict[str, tuple[_AccountedFigures, ...]],
) -> dict[
    tuple[str, tuple[str, ...], bool],
    tuple[_AccountedFigures, frozenset[str]],
]:
    """Index each unique origin-backed recipe by every entity exposing it."""
    mutable: dict[
        tuple[str, tuple[str, ...], bool],
        tuple[_AccountedFigures, set[str]],
    ] = {}
    for entity_id, records in states.items():
        for record in records:
            key = _accounted_key(record)
            existing = mutable.get(key)
            if existing is None:
                mutable[key] = (record, {entity_id})
            else:
                existing[1].add(entity_id)
    return {
        key: (record, frozenset(entity_ids))
        for key, (record, entity_ids) in mutable.items()
    }


def _eligible_contributor_key(
    contributor: _EligibleContributor,
) -> EligibleContributorKey:
    """Return the deterministic semantic identity of a contributor."""
    return (
        tuple(sorted(contributor.entity_ids)),
        _accounted_key(contributor.accounted),
        tuple(sorted(contributor.covered_units)),
        tuple(sorted(contributor.extras)),
    )


def _eligible_contributors(
    target_id: str,
    states: dict[str, tuple[_AccountedFigures, ...]],
    entities_by_id: dict[str, schema.Entity],
    structural_leaf_sets: dict[str, EntitySet],
    origin_units: dict[str, EntitySet],
    inclusive_ancestors: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
    shadowed_origin_ids: frozenset[str],
    population: dict[
        tuple[str, tuple[str, ...], bool],
        tuple[_AccountedFigures, frozenset[str]],
    ]
    | None = None,
) -> list[_EligibleContributor]:
    """Find contributors using graph branches and authoritative origin identity."""
    target_units = _scope_units(target_id, structural_leaf_sets)
    eligible: dict[
        tuple[str, tuple[str, ...], tuple[str, ...], bool],
        _EligibleContributor,
    ] = {}
    recipe_population = population or _provenance_population(states)
    for record, exposing_entity_ids in recipe_population.values():
        contributor_ids = exposing_entity_ids.difference({target_id})
        if not contributor_ids:
            continue
        if record.origins.issubset(shadowed_origin_ids):
            continue
        covered = _covered_target_units(
            record,
            target_units,
            origin_units,
            inclusive_descendants,
        )
        if not covered:
            continue
        extras = _record_extras(
            record,
            target_units,
            origin_units,
            inclusive_descendants,
        )
        if any(
            _extra_is_contested(
                target_id,
                extra_id,
                entities_by_id,
                inclusive_ancestors,
                inclusive_descendants,
            )
            for extra_id in extras
        ):
            continue

        semantic_key = (
            _figure_set_key(record.figures),
            tuple(sorted(covered)),
            tuple(sorted(extras)),
            record.sealed,
        )
        existing_contributor = eligible.get(semantic_key)
        if existing_contributor is None:
            eligible[semantic_key] = _EligibleContributor(
                entity_ids=frozenset(contributor_ids),
                accounted=record,
                covered_units=covered,
                extras=extras,
            )
        else:
            eligible[semantic_key] = _EligibleContributor(
                entity_ids=frozenset(
                    {*existing_contributor.entity_ids, *contributor_ids}
                ),
                accounted=_AccountedFigures(
                    existing_contributor.accounted.figures,
                    frozenset(
                        {*existing_contributor.accounted.origins, *record.origins}
                    ),
                    sealed=record.sealed,
                ),
                covered_units=existing_contributor.covered_units,
                extras=existing_contributor.extras,
            )

    contributors = list(eligible.values())
    dominated: set[int] = set()
    for index, contributor in enumerate(contributors):
        for other_index, other in enumerate(contributors):
            if index == other_index:
                continue
            if (
                contributor.covered_units == other.covered_units
                and contributor.extras == other.extras
                and other.accounted.sealed
                and contributor.accounted.origins < other.accounted.origins
            ):
                dominated.add(index)
                break
    return sorted(
        (
            contributor
            for index, contributor in enumerate(contributors)
            if index not in dominated
        ),
        key=_eligible_contributor_key,
    )


def _derivation_priority(extras: EntitySet) -> int:
    """Classify a derivation by whether represented branches remain outside."""
    return _PRIORITY_NO_EXTRAS if not extras else _PRIORITY_WITH_EXTRAS


def _single_complete_derivations(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    *,
    with_extras: bool,
) -> list[_Derivation]:
    """Return the strongest complete one-record copies before arithmetic."""
    candidates = [
        contributor
        for contributor in eligible
        if bool(contributor.extras) is with_extras
        and required.issubset(contributor.covered_units)
    ]
    sealed = [item for item in candidates if item.accounted.sealed]
    if sealed:
        candidates = sealed
    elif candidates:
        minimum_origins = min(len(item.accounted.origins) for item in candidates)
        candidates = [
            item
            for item in candidates
            if len(item.accounted.origins) == minimum_origins
        ]
    priority = _PRIORITY_RESIDUAL_COPY if with_extras else _PRIORITY_EXACT
    return [_Derivation(priority, item.accounted, item.extras) for item in candidates]


def _derivations_at_priority(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    priority: int,
    required_origin_ids: EntitySet = frozenset(),
) -> list[_Derivation]:
    """Search complete origin-disjoint derivations at one priority."""
    candidates = (
        [contributor for contributor in eligible if not contributor.extras]
        if priority == _PRIORITY_NO_EXTRAS
        else eligible
    )
    contributors_by_unit = {
        unit_id: [
            contributor
            for contributor in candidates
            if unit_id in contributor.covered_units
        ]
        for unit_id in required
    }
    contributors_by_origin = {
        origin_id: [
            contributor
            for contributor in candidates
            if origin_id in contributor.accounted.origins
        ]
        for origin_id in required_origin_ids
    }
    if (
        not candidates
        or not all(contributors_by_unit.values())
        or not all(contributors_by_origin.values())
    ):
        return []

    states: list[
        tuple[
            EntitySet,
            EntitySet,
            FigureSet | None,
            EntitySet,
            bool,
            tuple[EntitySet, ...],
            tuple[EntitySet, ...],
        ]
    ] = [(frozenset(), frozenset(), None, frozenset(), True, (), ())]
    visited: set[
        tuple[
            EntitySet,
            EntitySet,
            str | None,
            EntitySet,
            bool,
            tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
        ]
    ] = set()
    derivations: dict[
        tuple[tuple[str, tuple[str, ...], bool], tuple[str, ...]], _Derivation
    ] = {}

    while states:
        (
            covered,
            origins,
            figures,
            extras,
            sealed,
            selected_coverages,
            selected_origins,
        ) = states.pop()
        state_key = (
            covered,
            origins,
            _figure_set_key(figures) if figures is not None else None,
            extras,
            sealed,
            tuple(
                sorted(
                    (tuple(sorted(coverage)), tuple(sorted(selected_origin_ids)))
                    for coverage, selected_origin_ids in zip(
                        selected_coverages, selected_origins
                    )
                )
            ),
        )
        if state_key in visited:
            continue
        visited.add(state_key)

        if required.issubset(covered) and required_origin_ids.issubset(origins):
            if figures is None or _derivation_priority(extras) != priority:
                continue
            if any(
                required.issubset(
                    frozenset(
                        unit
                        for other_index, other_coverage in enumerate(selected_coverages)
                        if other_index != index
                        for unit in other_coverage
                    )
                )
                and required_origin_ids.issubset(
                    frozenset(
                        origin_id
                        for other_index, other_origin_ids in enumerate(selected_origins)
                        if other_index != index
                        for origin_id in other_origin_ids
                    )
                )
                for index in range(len(selected_coverages))
            ):
                continue
            record = _AccountedFigures(figures, origins, sealed=sealed)
            derivation = _Derivation(priority, record, extras)
            derivations[(_accounted_key(record), tuple(sorted(extras)))] = derivation
            continue

        uncovered = required.difference(covered)
        if uncovered:
            next_unit = min(
                uncovered,
                key=lambda unit_id: (len(contributors_by_unit[unit_id]), unit_id),
            )
            next_contributors = contributors_by_unit[next_unit]
        else:
            missing_origins = required_origin_ids.difference(origins)
            next_origin = min(
                missing_origins,
                key=lambda origin_id: (
                    len(contributors_by_origin[origin_id]),
                    origin_id,
                ),
            )
            next_contributors = contributors_by_origin[next_origin]

        for contributor in next_contributors:
            if origins.intersection(contributor.accounted.origins):
                continue
            next_figures = (
                contributor.accounted.figures
                if figures is None
                else _aggregate_figure_sets([figures, contributor.accounted.figures])
            )
            if next_figures is None:
                continue
            states.append(
                (
                    frozenset(covered | contributor.covered_units),
                    frozenset(origins | contributor.accounted.origins),
                    next_figures,
                    frozenset(extras | contributor.extras),
                    sealed and contributor.accounted.sealed,
                    (*selected_coverages, contributor.covered_units),
                    (*selected_origins, contributor.accounted.origins),
                )
            )

    return sorted(derivations.values(), key=lambda item: _accounted_key(item.accounted))


def _complete_derivations(
    eligible: list[_EligibleContributor], required: EntitySet
) -> list[_Derivation]:
    """Return only the highest-priority complete origin-based derivations."""
    if not required:
        return []
    exact_copy = _single_complete_derivations(eligible, required, with_extras=False)
    if exact_copy:
        return exact_copy
    no_extra = _derivations_at_priority(eligible, required, _PRIORITY_NO_EXTRAS)
    if no_extra:
        return no_extra
    residual_copy = _single_complete_derivations(eligible, required, with_extras=True)
    if residual_copy:
        return residual_copy
    return _derivations_at_priority(eligible, required, _PRIORITY_WITH_EXTRAS)


def _all_complete_derivations(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    required_origin_ids: EntitySet = frozenset(),
) -> list[_Derivation]:
    """Return every complete provenance recipe regardless of final priority."""
    if not required:
        return []
    derivations = [
        *[
            _Derivation(_PRIORITY_EXACT, contributor.accounted, contributor.extras)
            for contributor in eligible
            if not contributor.extras and required.issubset(contributor.covered_units)
        ],
        *_derivations_at_priority(
            eligible,
            required,
            _PRIORITY_NO_EXTRAS,
            required_origin_ids,
        ),
        *[
            _Derivation(
                _PRIORITY_RESIDUAL_COPY,
                contributor.accounted,
                contributor.extras,
            )
            for contributor in eligible
            if contributor.extras and required.issubset(contributor.covered_units)
        ],
        *_derivations_at_priority(
            eligible,
            required,
            _PRIORITY_WITH_EXTRAS,
            required_origin_ids,
        ),
    ]
    return list(
        {
            (
                derivation.priority,
                _accounted_key(derivation.accounted),
                tuple(sorted(derivation.extras)),
            ): derivation
            for derivation in derivations
        }.values()
    )


def _resolve_derivations(derivations: list[_Derivation]) -> _Resolution:
    """Apply category, extra-count, deduplication, and conflict semantics."""
    for priority in (
        _PRIORITY_EXACT,
        _PRIORITY_NO_EXTRAS,
        _PRIORITY_RESIDUAL_COPY,
        _PRIORITY_WITH_EXTRAS,
    ):
        at_priority = [item for item in derivations if item.priority == priority]
        if not at_priority:
            continue
        minimum_extra_count = min(len(item.extras) for item in at_priority)
        at_priority = [
            item for item in at_priority if len(item.extras) == minimum_extra_count
        ]
        figure_keys = {
            _figure_set_key(derivation.accounted.figures) for derivation in at_priority
        }
        if len(figure_keys) > 1:
            return _Resolution(conflicted=True)
        records_by_key = {
            _accounted_key(derivation.accounted): derivation.accounted
            for derivation in at_priority
        }
        return _Resolution(
            records=tuple(records_by_key[key] for key in sorted(records_by_key))
        )
    return _Resolution()


def _target_resolution(
    target_id: str,
    contributor_states: dict[str, tuple[_AccountedFigures, ...]],
    entities_by_id: dict[str, schema.Entity],
    structural_leaf_sets: dict[str, EntitySet],
    origin_units: dict[str, EntitySet],
    inclusive_ancestors: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
    shadowed_origin_ids: frozenset[str],
    population: dict[
        tuple[str, tuple[str, ...], bool],
        tuple[_AccountedFigures, frozenset[str]],
    ]
    | None = None,
    required_origin_ids: EntitySet = frozenset(),
) -> _Resolution:
    """Resolve one target while satisfying complete authoritative frontiers."""
    required = _required_target_units(
        target_id,
        structural_leaf_sets,
        origin_units,
        inclusive_descendants,
        shadowed_origin_ids,
    )
    eligible = _eligible_contributors(
        target_id,
        contributor_states,
        entities_by_id,
        structural_leaf_sets,
        origin_units,
        inclusive_ancestors,
        inclusive_descendants,
        shadowed_origin_ids,
        population,
    )
    derivations = (
        _all_complete_derivations(eligible, required, required_origin_ids)
        if required_origin_ids
        else _complete_derivations(eligible, required)
    )
    return _resolve_derivations(
        [
            derivation
            for derivation in derivations
            if required_origin_ids.issubset(derivation.accounted.origins)
        ]
    )


def _discover_provenance_recipes(
    states: dict[str, tuple[_AccountedFigures, ...]],
    dynamic_target_ids: list[str],
    conflicted_frontiers: dict[str, EntitySet],
    entities_by_id: dict[str, schema.Entity],
    structural_leaf_sets: dict[str, EntitySet],
    origin_units: dict[str, EntitySet],
    inclusive_ancestors: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
    shadowed_origin_ids: frozenset[str],
) -> None:
    """Discover every origin-backed recipe before resolving any target."""
    for _pass in range(max(1, len(dynamic_target_ids))):
        snapshot = dict(states)
        snapshot_population = _provenance_population(snapshot)
        known_recipe_keys = set(snapshot_population)
        additions: dict[
            str, dict[tuple[str, tuple[str, ...], bool], _AccountedFigures]
        ] = {}
        for target_id in dynamic_target_ids:
            frontier_requirements = _frontier_requirements(
                target_id,
                conflicted_frontiers,
                structural_leaf_sets,
            )
            if frontier_requirements.conflicted:
                continue
            required = _required_target_units(
                target_id,
                structural_leaf_sets,
                origin_units,
                inclusive_descendants,
                shadowed_origin_ids,
            )
            eligible = _eligible_contributors(
                target_id,
                snapshot,
                entities_by_id,
                structural_leaf_sets,
                origin_units,
                inclusive_ancestors,
                inclusive_descendants,
                shadowed_origin_ids,
                snapshot_population,
            )
            existing = {_accounted_key(record) for record in states[target_id]}
            for derivation in _all_complete_derivations(
                eligible,
                required,
                frontier_requirements.origins,
            ):
                if not frontier_requirements.origins.issubset(
                    derivation.accounted.origins
                ):
                    continue
                key = _accounted_key(derivation.accounted)
                if key not in existing:
                    additions.setdefault(target_id, {})[key] = derivation.accounted

        if not additions:
            return
        introduced_new_recipe = any(
            key not in known_recipe_keys
            for records_by_key in additions.values()
            for key in records_by_key
        )
        for target_id, records_by_key in additions.items():
            states[target_id] = tuple(
                {
                    _accounted_key(record): record
                    for record in (*states[target_id], *records_by_key.values())
                }[key]
                for key in sorted(
                    {
                        _accounted_key(record)
                        for record in (*states[target_id], *records_by_key.values())
                    }
                )
            )
        if not introduced_new_recipe:
            return


def _breakdown_signature(breakdown: schema.Breakdown) -> BreakdownSignature:
    """Return an order-independent signature for one complete breakdown."""
    return (
        breakdown.entity_id,
        breakdown.id,
        _canonical_json(breakdown.qualifiers),
        tuple(
            sorted(_canonical_metric_signature(metric) for metric in breakdown.metrics)
        ),
    )


def _deduplicate_breakdowns(
    breakdowns: list[schema.Breakdown],
) -> list[schema.Breakdown]:
    """Keep the first occurrence of each semantically identical breakdown."""
    seen: set[BreakdownSignature] = set()
    unique: list[schema.Breakdown] = []
    for breakdown in breakdowns:
        signature = _breakdown_signature(breakdown)
        if signature not in seen:
            seen.add(signature)
            unique.append(breakdown)
    return unique


def _projected_breakdowns(
    entity_id: str, figure_set: FigureSet
) -> list[schema.Breakdown]:
    """Materialize canonical figures for a target without exposing provenance."""
    return [
        schema.Breakdown(
            entity_id=entity_id,
            metrics=[copy.deepcopy(metric) for metric in figure.metrics],
            qualifiers=copy.deepcopy(figure.qualifiers),
        )
        for figure in figure_set
    ]


def _project_entity_breakdowns(result: schema.Result) -> schema.Result:
    """Project authoritative figures through explicit entity relationships.

    Existing breakdowns remain authoritative. Every usable authoritative record
    carries the ID of the entity that owns the original metric. Projection copies
    one complete origin unchanged or combines distinct origins only through
    compatible explicit ``sum`` metrics.

    Declared direct authoritative parents form a complete nearest frontier. All
    such parents must be compatible, and aggregate authoritative ancestors are
    shadowed by nearer authoritative descendants. Subsequent inference follows
    graph branches and preserves origin IDs, so downstream code-stack and module
    entities can combine a resolved frontier without reconsidering its individual
    constituent records.

    Structural terminal leaves determine complete target coverage and contested
    extras. Authoritative entity IDs remain the contribution identities, so
    distinct measurements can overlap structurally without being mistaken for
    duplicate accounting origins.

    The input result is never mutated. Duplicate existing breakdowns are removed
    in first-seen order and projected breakdowns are appended deterministically.
    """
    graph = _validated_graph(result)
    entities_by_id = {entity.id: entity for entity in result.entities}
    frontier_kind_edges = _declared_frontier_kind_edges(result.entity_kinds)
    structural_leaf_sets = graph.structural_leaf_sets()
    inclusive_ancestors = graph.inclusive_ancestors()
    inclusive_descendants = _inclusive_descendants(graph)
    authoritative_breakdowns = _deduplicate_breakdowns(result.breakdowns)
    authoritative_entity_ids = {
        breakdown.entity_id for breakdown in authoritative_breakdowns
    }
    origin_units = _origin_units(authoritative_entity_ids, structural_leaf_sets)
    states, _invalid_origin_ids = _authoritative_sources(authoritative_breakdowns)
    shadowed_origin_ids = _shadowed_authoritative_origins(
        graph,
        authoritative_entity_ids,
        frontier_kind_edges,
        entities_by_id,
    )
    direct_resolutions = _direct_authoritative_parent_resolutions(
        graph,
        states,
        authoritative_entity_ids,
        frontier_kind_edges,
        entities_by_id,
        structural_leaf_sets,
        origin_units,
        inclusive_ancestors,
        inclusive_descendants,
        shadowed_origin_ids,
    )
    conflicted_frontiers = {
        entity_id: resolution.required_origins
        for entity_id, resolution in direct_resolutions.items()
        if resolution.conflicted
    }

    dynamic_target_ids: list[str] = []
    for entity_id in sorted(entities_by_id):
        if entity_id in authoritative_entity_ids:
            continue
        direct = direct_resolutions.get(entity_id)
        if direct is not None:
            states[entity_id] = (
                (direct.record,)
                if direct.record is not None and not direct.conflicted
                else ()
            )
        else:
            states[entity_id] = ()
            dynamic_target_ids.append(entity_id)

    _discover_provenance_recipes(
        states,
        dynamic_target_ids,
        conflicted_frontiers,
        entities_by_id,
        structural_leaf_sets,
        origin_units,
        inclusive_ancestors,
        inclusive_descendants,
        shadowed_origin_ids,
    )

    projected: list[schema.Breakdown] = []
    dynamic_targets = set(dynamic_target_ids)
    final_population = _provenance_population(states)
    for entity_id in sorted(entities_by_id):
        if entity_id in authoritative_entity_ids:
            continue
        if entity_id in dynamic_targets:
            frontier_requirements = _frontier_requirements(
                entity_id,
                conflicted_frontiers,
                structural_leaf_sets,
            )
            if frontier_requirements.conflicted:
                continue
            records = _target_resolution(
                entity_id,
                states,
                entities_by_id,
                structural_leaf_sets,
                origin_units,
                inclusive_ancestors,
                inclusive_descendants,
                shadowed_origin_ids,
                final_population,
                frontier_requirements.origins,
            ).records
        else:
            records = states.get(entity_id, ())
        if records:
            projected.extend(_projected_breakdowns(entity_id, records[0].figures))

    if not projected and len(authoritative_breakdowns) == len(result.breakdowns):
        return result
    return replace(result, breakdowns=[*authoritative_breakdowns, *projected])


def project_entity_breakdowns(result: schema.Result) -> schema.Result:
    """Project breakdowns from the complete authoritative provenance closure."""
    return _project_entity_breakdowns(result)
