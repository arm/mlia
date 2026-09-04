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
origins. Recipes capable of feeding another target are discovered to a fixed
point before final resolution, and every recipe retains the authoritative IDs
and arithmetic that produced it. Final resolution therefore compares the
relevant provenance closure instead of depending on traversal order or
prematurely committed values.

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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
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
AccountedKey = tuple[str, tuple[str, ...], bool]
CoverageCache = dict[tuple[str, AccountedKey], tuple[EntitySet, EntitySet]]

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
    figure_key: str = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Cache the canonical figure representation once per record."""
        object.__setattr__(self, "figure_key", _figure_set_key(self.figures))


@dataclass(frozen=True)
class _EligibleContributor:
    """Semantically equivalent target-compatible figure sources."""

    entity_ids: frozenset[str]
    accounted: _AccountedFigures
    covered_units: EntitySet
    extras: EntitySet


@dataclass(frozen=True)
class _IndexedContributor:
    """One contributor encoded against a fixed derivation-search universe."""

    contributor: _EligibleContributor
    coverage_mask: int
    origin_mask: int
    extra_mask: int
    figure_key: str


@dataclass(frozen=True)
class _DerivationSearchIndex:
    """Precomputed bitmask indexes for one derivation search."""

    contributors: tuple[_IndexedContributor, ...]
    unit_options: tuple[tuple[str, int, tuple[_IndexedContributor, ...]], ...]
    origin_options: tuple[tuple[str, int, tuple[_IndexedContributor, ...]], ...]
    required_mask: int
    required_origin_mask: int
    origin_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]


@dataclass(frozen=True)
class _DerivationState:
    """One queued derivation state using the search index's bit universes."""

    covered_mask: int
    origin_mask: int
    figures: FigureSet | None
    figure_key: str | None
    extra_mask: int
    sealed: bool
    selected: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _DirectAtom:
    """One directly represented, indivisible authoritative origin family."""

    origin_mask: int
    coverage_mask: int
    extra_mask: int
    record: _AccountedFigures


@dataclass(frozen=True)
class _DirectAtomSpace:
    """A proven direct-atom decomposition for one filtered final search."""

    atoms: tuple[_DirectAtom, ...]
    required_atom_mask: int


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
        record.figure_key,
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
) -> EntitySet:
    """Return terminal target branches represented by at least one origin."""
    return target_units.intersection(_record_origin_units(record, origin_units))


def _record_extras(
    record: _AccountedFigures,
    target_units: EntitySet,
    origin_units: dict[str, EntitySet],
) -> EntitySet:
    """Return represented terminal branches outside the target hierarchy."""
    return _record_origin_units(record, origin_units).difference(target_units)


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
    shadowed_origin_ids: EntitySet,
) -> EntitySet:
    """Return target leaves covered by non-shadowed authoritative origins."""
    all_origin_units = frozenset(
        unit_id
        for origin_id, units in origin_units.items()
        if origin_id not in shadowed_origin_ids
        for unit_id in units
    )
    return _scope_units(target_id, structural_leaf_sets).intersection(all_origin_units)


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
            ):
                conflicted = True
                break
            extras = _record_extras(
                record,
                target_units,
                origin_units,
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
        AccountedKey,
        tuple[_AccountedFigures, frozenset[str]],
    ]
    | None = None,
    coverage_cache: CoverageCache | None = None,
    contested_cache: dict[tuple[str, str], bool] | None = None,
) -> list[_EligibleContributor]:
    """Find contributors using graph branches and authoritative origin identity."""
    target_units = _scope_units(target_id, structural_leaf_sets)
    eligible: dict[
        tuple[str, tuple[str, ...], tuple[str, ...], bool],
        _EligibleContributor,
    ] = {}
    recipe_population = population or _provenance_population(states)
    for record_key, (record, exposing_entity_ids) in recipe_population.items():
        contributor_ids = exposing_entity_ids.difference({target_id})
        if not contributor_ids:
            continue
        if record.origins.issubset(shadowed_origin_ids):
            continue
        cache_key = (target_id, record_key)
        cached_coverage = (
            coverage_cache.get(cache_key) if coverage_cache is not None else None
        )
        if cached_coverage is None:
            covered = _covered_target_units(
                record,
                target_units,
                origin_units,
            )
            extras = _record_extras(
                record,
                target_units,
                origin_units,
            )
            if coverage_cache is not None:
                coverage_cache[cache_key] = covered, extras
        else:
            covered, extras = cached_coverage
        if not covered:
            continue
        contested = False
        for extra_id in extras:
            contested_key = (target_id, extra_id)
            extra_contested = (
                contested_cache.get(contested_key)
                if contested_cache is not None
                else None
            )
            if extra_contested is None:
                extra_contested = _extra_is_contested(
                    target_id,
                    extra_id,
                    entities_by_id,
                    inclusive_ancestors,
                    inclusive_descendants,
                )
                if contested_cache is not None:
                    contested_cache[contested_key] = extra_contested
            if extra_contested:
                contested = True
                break
        if contested:
            continue

        semantic_key = (
            record_key[0],
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


def _entity_mask(entity_ids: EntitySet, positions: dict[str, int]) -> int:
    """Encode entity IDs using a fixed search-local bit universe."""
    mask = 0
    for entity_id in entity_ids:
        mask |= 1 << positions[entity_id]
    return mask


def _entities_from_mask(entity_ids: tuple[str, ...], mask: int) -> EntitySet:
    """Decode a search-local bitmask into entity IDs."""
    return frozenset(
        entity_id for index, entity_id in enumerate(entity_ids) if mask & (1 << index)
    )


def _derivation_search_index(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    priority: int,
    required_origin_ids: EntitySet,
) -> _DerivationSearchIndex | None:
    """Build fixed bitmask indexes before exploring derivation states."""
    candidates = (
        [contributor for contributor in eligible if not contributor.extras]
        if priority == _PRIORITY_NO_EXTRAS
        else eligible
    )
    if not candidates:
        return None

    unit_ids = tuple(sorted(required))
    origin_ids = tuple(
        sorted(
            required_origin_ids.union(
                origin_id
                for contributor in candidates
                for origin_id in contributor.accounted.origins
            )
        )
    )
    extra_ids = tuple(
        sorted(
            {extra_id for contributor in candidates for extra_id in contributor.extras}
        )
    )
    unit_positions = {entity_id: index for index, entity_id in enumerate(unit_ids)}
    origin_positions = {entity_id: index for index, entity_id in enumerate(origin_ids)}
    extra_positions = {entity_id: index for index, entity_id in enumerate(extra_ids)}
    indexed = tuple(
        _IndexedContributor(
            contributor=contributor,
            coverage_mask=_entity_mask(
                contributor.covered_units.intersection(required), unit_positions
            ),
            origin_mask=_entity_mask(contributor.accounted.origins, origin_positions),
            extra_mask=_entity_mask(contributor.extras, extra_positions),
            figure_key=contributor.accounted.figure_key,
        )
        for contributor in candidates
    )

    unit_options = tuple(
        sorted(
            [
                (
                    unit_id,
                    unit_bit,
                    tuple(
                        contributor
                        for contributor in indexed
                        if contributor.coverage_mask & unit_bit
                    ),
                )
                for unit_id, unit_index in unit_positions.items()
                for unit_bit in (1 << unit_index,)
            ],
            key=lambda item: (len(item[2]), item[0]),
        )
    )
    origin_options = tuple(
        sorted(
            [
                (
                    origin_id,
                    origin_bit,
                    tuple(
                        contributor
                        for contributor in indexed
                        if contributor.origin_mask & origin_bit
                    ),
                )
                for origin_id in required_origin_ids
                for origin_bit in (1 << origin_positions[origin_id],)
            ],
            key=lambda item: (len(item[2]), item[0]),
        )
    )
    if any(not options for _entity_id, _bit, options in unit_options) or any(
        not options for _entity_id, _bit, options in origin_options
    ):
        return None

    return _DerivationSearchIndex(
        contributors=indexed,
        unit_options=unit_options,
        origin_options=origin_options,
        required_mask=(1 << len(unit_ids)) - 1,
        required_origin_mask=_entity_mask(required_origin_ids, origin_positions),
        origin_ids=origin_ids,
        extra_ids=extra_ids,
    )


def _derivation_state_key(
    state: _DerivationState,
) -> tuple[int, int, str | None, int, bool, tuple[tuple[int, int], ...]]:
    """Return the semantic identity used to deduplicate queued states."""
    return (
        state.covered_mask,
        state.origin_mask,
        state.figure_key,
        state.extra_mask,
        state.sealed,
        state.selected,
    )


def _selection_is_redundant(
    selected: tuple[tuple[int, int], ...],
    required_mask: int,
    required_origin_mask: int,
) -> bool:
    """Return whether removing one selected contributor keeps requirements met."""
    coverage_prefix = [0]
    origin_prefix = [0]
    for coverage_mask, origin_mask in selected:
        coverage_prefix.append(coverage_prefix[-1] | coverage_mask)
        origin_prefix.append(origin_prefix[-1] | origin_mask)

    coverage_suffix = 0
    origin_suffix = 0
    for index in range(len(selected) - 1, -1, -1):
        if (coverage_prefix[index] | coverage_suffix) == required_mask and (
            origin_prefix[index] | origin_suffix
        ) & required_origin_mask == required_origin_mask:
            return True
        coverage_suffix |= selected[index][0]
        origin_suffix |= selected[index][1]
    return False


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


def _set_bit_indexes(mask: int) -> Iterator[int]:
    """Yield set-bit positions from least to most significant."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _minimal_masks(masks: Iterable[int]) -> tuple[int, ...]:
    """Return unique masks with no smaller input mask as a subset."""
    minimal: list[int] = []
    for mask in sorted(set(masks), key=lambda item: (item.bit_count(), item)):
        if any(previous & mask == previous for previous in minimal):
            continue
        minimal.append(mask)
    return tuple(minimal)


def _minimum_extra_masks(search: _DerivationSearchIndex) -> Iterator[int]:
    """Yield satisfying masks with the fewest extra IDs using best-first search."""
    constraints = tuple(
        masks
        for _entity_id, _bit, options in (
            *search.unit_options,
            *search.origin_options,
        )
        if 0 not in (masks := _minimal_masks(item.extra_mask for item in options))
    )
    queue = [(0, 0)]
    enqueued = {0}
    best_count: int | None = None
    while queue:
        count, mask = heappop(queue)
        if best_count is not None and count > best_count:
            return
        unsatisfied = tuple(
            options
            for options in constraints
            if not any(option & mask == option for option in options)
        )
        if not unsatisfied:
            best_count = count
            yield mask
            continue

        options = min(unsatisfied, key=lambda item: (len(item), item))
        for option in options:
            next_mask = mask | option
            if next_mask in enqueued:
                continue
            enqueued.add(next_mask)
            heappush(queue, (next_mask.bit_count(), next_mask))


def _aggregate_atom_records(
    atoms: tuple[_DirectAtom, ...],
    atom_mask: int,
) -> _AccountedFigures | None:
    """Aggregate one structurally selected origin family with normal semantics."""
    return _aggregate_records(
        [atoms[position].record for position in _set_bit_indexes(atom_mask)]
    )


def _direct_atom_space(
    search: _DerivationSearchIndex,
    allowed_extra_mask: int,
) -> _DirectAtomSpace | None:
    """Prove that filtered contributors decompose into direct origin atoms."""
    candidates = tuple(
        contributor
        for contributor in search.contributors
        if not contributor.extra_mask & ~allowed_extra_mask
    )
    coverage_union = 0
    origin_union = 0
    for contributor in candidates:
        coverage_union |= contributor.coverage_mask
        origin_union |= contributor.origin_mask
    if (
        coverage_union != search.required_mask
        or (origin_union & search.required_origin_mask) != search.required_origin_mask
    ):
        return None

    signatures: dict[tuple[int, ...], int] = {}
    for origin_position in _set_bit_indexes(origin_union):
        origin_bit = 1 << origin_position
        signature = tuple(
            index
            for index, contributor in enumerate(candidates)
            if contributor.origin_mask & origin_bit
        )
        signatures[signature] = signatures.get(signature, 0) | origin_bit
    origin_masks = tuple(sorted(signatures.values()))

    atom_positions: dict[int, int] = {}
    required_atom_mask = 0
    for atom_position, origin_mask in enumerate(origin_masks):
        required_part = origin_mask & search.required_origin_mask
        if required_part and required_part != origin_mask:
            return None
        if required_part:
            required_atom_mask |= 1 << atom_position
        for origin_position in _set_bit_indexes(origin_mask):
            atom_positions[origin_position] = atom_position

    contributor_atom_masks: list[int] = []
    direct_rows: dict[int, list[_IndexedContributor]] = {}
    for contributor in candidates:
        atom_mask = 0
        for origin_position in _set_bit_indexes(contributor.origin_mask):
            atom_mask |= 1 << atom_positions[origin_position]
        contributor_atom_masks.append(atom_mask)
        if atom_mask.bit_count() == 1:
            direct_rows.setdefault(atom_mask.bit_length() - 1, []).append(contributor)

    atoms: list[_DirectAtom] = []
    for atom_position, origin_mask in enumerate(origin_masks):
        rows = direct_rows.get(atom_position)
        if not rows:
            return None
        semantics = {
            (row.coverage_mask, row.extra_mask, row.figure_key) for row in rows
        }
        if len(semantics) != 1:
            return None
        coverage_mask, extra_mask, _figure_key = next(iter(semantics))
        preferred = max(rows, key=lambda row: row.contributor.accounted.sealed)
        atoms.append(
            _DirectAtom(
                origin_mask,
                coverage_mask,
                extra_mask,
                preferred.contributor.accounted,
            )
        )
    atom_tuple = tuple(atoms)

    aggregate_cache: dict[int, _AccountedFigures | None] = {}
    for contributor, atom_mask in zip(candidates, contributor_atom_masks):
        aggregate = aggregate_cache.get(atom_mask)
        if atom_mask not in aggregate_cache:
            aggregate = _aggregate_atom_records(atom_tuple, atom_mask)
            aggregate_cache[atom_mask] = aggregate
        if aggregate is None:
            return None
        coverage_mask = 0
        extra_mask = 0
        for atom_position in _set_bit_indexes(atom_mask):
            coverage_mask |= atom_tuple[atom_position].coverage_mask
            extra_mask |= atom_tuple[atom_position].extra_mask
        if (
            coverage_mask != contributor.coverage_mask
            or extra_mask != contributor.extra_mask
            or aggregate.origins != contributor.contributor.accounted.origins
            or aggregate.figure_key != contributor.figure_key
        ):
            return None

    return _DirectAtomSpace(atom_tuple, required_atom_mask)


def _forced_complete_atom_mask(
    space: _DirectAtomSpace,
    required_coverage_mask: int,
) -> int | None:
    """Return the unique complete family only when every atom is forced."""
    atom_mask = space.required_atom_mask
    coverage_mask = 0
    for position in _set_bit_indexes(atom_mask):
        coverage_mask |= space.atoms[position].coverage_mask

    while coverage_mask != required_coverage_mask:
        missing_mask = required_coverage_mask & ~coverage_mask
        forced_atoms = 0
        for unit_position in _set_bit_indexes(missing_mask):
            unit_bit = 1 << unit_position
            options = tuple(
                1 << atom_position
                for atom_position, atom in enumerate(space.atoms)
                if not atom_mask & (1 << atom_position)
                and atom.coverage_mask & unit_bit
            )
            if len(options) != 1:
                return None
            forced_atoms |= options[0]
        if not forced_atoms:
            return None
        atom_mask |= forced_atoms
        for position in _set_bit_indexes(forced_atoms):
            coverage_mask |= space.atoms[position].coverage_mask

    all_atoms_mask = (1 << len(space.atoms)) - 1
    return atom_mask if atom_mask == all_atoms_mask else None


def _atom_addition_is_exact(atoms: tuple[_DirectAtom, ...]) -> bool:
    """Prove every atom-value addition order is exactly representable."""
    ratios: list[tuple[int, int]] = []
    for atom in atoms:
        for figure in atom.record.figures:
            for metric in figure.metrics:
                value = metric.value
                if type(value) not in (int, float):
                    return False
                numeric_value = cast(float | int, value)
                if isinstance(numeric_value, float):
                    if not math.isfinite(numeric_value):
                        return False
                    ratios.append(numeric_value.as_integer_ratio())
                else:
                    ratios.append((numeric_value, 1))

    common_denominator = max((denominator for _value, denominator in ratios), default=1)
    absolute_coefficients = sum(
        abs(value) * (common_denominator // denominator)
        for value, denominator in ratios
    )
    return absolute_coefficients <= _MAX_SAFE_INTEGER


def _direct_atom_derivations(
    search: _DerivationSearchIndex,
    priority: int,
) -> list[_Derivation] | None:
    """Resolve covers whose contributors exactly reconstruct from direct atoms."""
    extra_masks: Iterable[int] = (
        (0,) if priority == _PRIORITY_NO_EXTRAS else _minimum_extra_masks(search)
    )
    completed: dict[tuple[AccountedKey, tuple[str, ...]], _Derivation] = {}
    for allowed_extra_mask in extra_masks:
        space = _direct_atom_space(search, allowed_extra_mask)
        if space is None or not _atom_addition_is_exact(space.atoms):
            return None
        atom_mask = _forced_complete_atom_mask(space, search.required_mask)
        if atom_mask is None:
            return None
        record = _aggregate_atom_records(space.atoms, atom_mask)
        if record is None:
            return None
        extra_mask = 0
        for position in _set_bit_indexes(atom_mask):
            extra_mask |= space.atoms[position].extra_mask
        if bool(extra_mask) is not (priority == _PRIORITY_WITH_EXTRAS):
            return None
        extras = _entities_from_mask(search.extra_ids, extra_mask)
        derivation = _Derivation(priority, record, extras)
        completed[(_accounted_key(record), tuple(sorted(extras)))] = derivation

    return list(completed.values()) if completed else None


def _iter_derivations_at_priority_reference(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    priority: int,
    required_origin_ids: EntitySet = frozenset(),
) -> Iterator[_Derivation]:
    """Yield derivations using the general contributor-selection search."""
    search = _derivation_search_index(eligible, required, priority, required_origin_ids)
    if search is None:
        return

    initial = _DerivationState(0, 0, None, None, 0, True, ())
    states = [initial]
    enqueued = {_derivation_state_key(initial)}
    aggregation_cache: dict[tuple[str, str], tuple[FigureSet, str] | None] = {}
    yielded: set[tuple[AccountedKey, tuple[str, ...]]] = set()

    while states:
        state = states.pop()
        if (
            state.covered_mask == search.required_mask
            and state.origin_mask & search.required_origin_mask
            == search.required_origin_mask
        ):
            if state.figures is None or bool(state.extra_mask) is not (
                priority == _PRIORITY_WITH_EXTRAS
            ):
                continue
            if _selection_is_redundant(
                state.selected,
                search.required_mask,
                search.required_origin_mask,
            ):
                continue
            origins = _entities_from_mask(search.origin_ids, state.origin_mask)
            extras = _entities_from_mask(search.extra_ids, state.extra_mask)
            record = _AccountedFigures(state.figures, origins, sealed=state.sealed)
            derivation_key = (_accounted_key(record), tuple(sorted(extras)))
            if derivation_key not in yielded:
                yielded.add(derivation_key)
                yield _Derivation(priority, record, extras)
            continue

        if state.covered_mask != search.required_mask:
            next_contributors = next(
                options
                for _entity_id, bit, options in search.unit_options
                if not state.covered_mask & bit
            )
        else:
            next_contributors = next(
                options
                for _entity_id, bit, options in search.origin_options
                if not state.origin_mask & bit
            )

        for indexed in next_contributors:
            if state.origin_mask & indexed.origin_mask:
                continue
            if state.figures is None:
                next_figures = indexed.contributor.accounted.figures
                next_figure_key = indexed.figure_key
            else:
                assert state.figure_key is not None
                aggregation_key = (state.figure_key, indexed.figure_key)
                if aggregation_key not in aggregation_cache:
                    aggregated = _aggregate_figure_sets(
                        [state.figures, indexed.contributor.accounted.figures]
                    )
                    aggregation_cache[aggregation_key] = (
                        (aggregated, _figure_set_key(aggregated))
                        if aggregated is not None
                        else None
                    )
                aggregation = aggregation_cache[aggregation_key]
                if aggregation is None:
                    continue
                next_figures, next_figure_key = aggregation

            selected = tuple(
                sorted(
                    (
                        *state.selected,
                        (indexed.coverage_mask, indexed.origin_mask),
                    )
                )
            )
            next_state = _DerivationState(
                covered_mask=state.covered_mask | indexed.coverage_mask,
                origin_mask=state.origin_mask | indexed.origin_mask,
                figures=next_figures,
                figure_key=next_figure_key,
                extra_mask=state.extra_mask | indexed.extra_mask,
                sealed=state.sealed and indexed.contributor.accounted.sealed,
                selected=selected,
            )
            state_key = _derivation_state_key(next_state)
            if state_key in enqueued:
                continue
            enqueued.add(state_key)
            states.append(next_state)


def _iter_derivations_at_priority(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    priority: int,
    required_origin_ids: EntitySet = frozenset(),
) -> Iterator[_Derivation]:
    """Yield final derivations through a guarded structural fast path."""
    if required_origin_ids:
        search = _derivation_search_index(
            eligible,
            required,
            priority,
            required_origin_ids,
        )
        if search is None:
            return
        optimized = _direct_atom_derivations(search, priority)
        if optimized is not None:
            yield from optimized
            return
    yield from _iter_derivations_at_priority_reference(
        eligible,
        required,
        priority,
        required_origin_ids,
    )


def _derivations_at_priority(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    priority: int,
    required_origin_ids: EntitySet = frozenset(),
) -> list[_Derivation]:
    """Return all reference derivations needed for recipe discovery."""
    return sorted(
        _iter_derivations_at_priority_reference(
            eligible,
            required,
            priority,
            required_origin_ids,
        ),
        key=lambda item: _accounted_key(item.accounted),
    )


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


def _copy_derivations(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    *,
    with_extras: bool,
    required_origin_ids: EntitySet,
) -> list[_Derivation]:
    """Return complete copies, applying frontier-specific origin requirements."""
    if not required_origin_ids:
        return _single_complete_derivations(
            eligible,
            required,
            with_extras=with_extras,
        )

    priority = _PRIORITY_RESIDUAL_COPY if with_extras else _PRIORITY_EXACT
    return [
        _Derivation(priority, contributor.accounted, contributor.extras)
        for contributor in eligible
        if bool(contributor.extras) is with_extras
        and required.issubset(contributor.covered_units)
        and required_origin_ids.issubset(contributor.accounted.origins)
    ]


def _resolve_priority_derivations(
    derivations: Iterable[_Derivation],
    minimum_possible_extra_count: int,
) -> _Resolution:
    """Resolve one priority, stopping once its strongest outcome conflicts."""
    best_extra_count: int | None = None
    best_figure_key: str | None = None
    best_record: _AccountedFigures | None = None
    conflicted = False

    for derivation in derivations:
        extra_count = len(derivation.extras)
        figure_key = derivation.accounted.figure_key
        if best_extra_count is None or extra_count < best_extra_count:
            best_extra_count = extra_count
            best_figure_key = figure_key
            best_record = derivation.accounted
            conflicted = False
        elif extra_count > best_extra_count:
            continue
        elif figure_key != best_figure_key:
            conflicted = True

        if conflicted and best_extra_count == minimum_possible_extra_count:
            return _Resolution(conflicted=True)

    if conflicted:
        return _Resolution(conflicted=True)
    return (
        _Resolution(records=(best_record,))
        if best_record is not None
        else _Resolution()
    )


def _eligible_resolution(
    eligible: list[_EligibleContributor],
    required: EntitySet,
    required_origin_ids: EntitySet,
) -> _Resolution:
    """Resolve already selected contributors in normal priority order."""
    searches: tuple[tuple[Iterable[_Derivation], int], ...] = (
        (
            _copy_derivations(
                eligible,
                required,
                with_extras=False,
                required_origin_ids=required_origin_ids,
            ),
            0,
        ),
        (
            _iter_derivations_at_priority(
                eligible,
                required,
                _PRIORITY_NO_EXTRAS,
                required_origin_ids,
            ),
            0,
        ),
        (
            _copy_derivations(
                eligible,
                required,
                with_extras=True,
                required_origin_ids=required_origin_ids,
            ),
            1,
        ),
        (
            _iter_derivations_at_priority(
                eligible,
                required,
                _PRIORITY_WITH_EXTRAS,
                required_origin_ids,
            ),
            1,
        ),
    )
    for derivations, minimum_extra_count in searches:
        resolution = _resolve_priority_derivations(
            derivations,
            minimum_extra_count,
        )
        if resolution.records or resolution.conflicted:
            return resolution
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
        shadowed_origin_ids,
    )
    if not required:
        return _Resolution()

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
    return _eligible_resolution(eligible, required, required_origin_ids)


def _recipe_source_ids(
    states: dict[str, tuple[_AccountedFigures, ...]],
    dynamic_target_ids: list[str],
    conflicted_frontiers: dict[str, EntitySet],
    structural_leaf_sets: dict[str, EntitySet],
    origin_units: dict[str, EntitySet],
    entities_by_id: dict[str, schema.Entity],
    inclusive_ancestors: dict[str, EntitySet],
    inclusive_descendants: dict[str, EntitySet],
    shadowed_origin_ids: EntitySet,
) -> list[str]:
    """Select recipe producers and expose sealed recipes to equal-scope peers."""
    required_cache: dict[str, EntitySet] = {}
    eligible_cache: dict[str, list[_EligibleContributor]] = {}
    reachable_cache: dict[str, EntitySet] = {}
    contested_units_cache: dict[str, EntitySet] = {}
    coverage_cache: CoverageCache = {}
    extra_contested_cache: dict[tuple[str, str], bool] = {}
    population = _provenance_population(states)

    def required_units(target_id: str) -> EntitySet:
        required = required_cache.get(target_id)
        if required is None:
            required = _required_target_units(
                target_id,
                structural_leaf_sets,
                origin_units,
                shadowed_origin_ids,
            )
            required_cache[target_id] = required
        return required

    def eligible_contributors(target_id: str) -> list[_EligibleContributor]:
        eligible = eligible_cache.get(target_id)
        if eligible is None:
            eligible = _eligible_contributors(
                target_id,
                states,
                entities_by_id,
                structural_leaf_sets,
                origin_units,
                inclusive_ancestors,
                inclusive_descendants,
                frozenset(shadowed_origin_ids),
                population,
                coverage_cache,
                extra_contested_cache,
            )
            eligible_cache[target_id] = eligible
        return eligible

    def reachable_units(target_id: str) -> EntitySet:
        reachable = reachable_cache.get(target_id)
        if reachable is None:
            required = required_units(target_id)
            eligible = eligible_contributors(target_id)
            covered = frozenset(
                unit_id
                for contributor in eligible
                for unit_id in contributor.covered_units
            )
            reachable = (
                frozenset(
                    unit_id
                    for contributor in eligible
                    for unit_id in contributor.covered_units.union(contributor.extras)
                )
                if required.issubset(covered)
                else frozenset()
            )
            reachable_cache[target_id] = reachable
        return reachable

    def contested_units(target_id: str) -> EntitySet:
        contested = contested_units_cache.get(target_id)
        if contested is None:
            target = entities_by_id[target_id]
            target_ancestors = inclusive_ancestors[target_id]
            contested = frozenset(
                unit_id
                for other_id, other in entities_by_id.items()
                if other_id != target_id
                and (
                    other.kind == target.kind
                    or bool(
                        target_ancestors.intersection(inclusive_ancestors[other_id])
                    )
                )
                for unit_id in structural_leaf_sets[other_id]
            )
            contested_units_cache[target_id] = contested
        return contested

    recipe_source_ids: list[str] = []
    peers_by_required: dict[EntitySet, list[str]] = {}
    for producer_id in dynamic_target_ids:
        producer_required = required_units(producer_id)
        if not producer_required:
            continue
        peers_by_required.setdefault(producer_required, []).append(producer_id)
        producer_reachable = reachable_units(producer_id)
        for consumer_id in dynamic_target_ids:
            if consumer_id == producer_id:
                continue
            consumer_required = required_units(consumer_id)
            if consumer_required == producer_required:
                continue
            consumer_scope = structural_leaf_sets[consumer_id]
            if not producer_reachable.intersection(consumer_scope):
                continue
            outside_consumer = producer_required.difference(consumer_scope)
            if outside_consumer.intersection(contested_units(consumer_id)):
                continue
            recipe_source_ids.append(producer_id)
            break

    peer_additions: dict[str, _AccountedFigures] = {}
    for peer_ids in peers_by_required.values():
        if len(peer_ids) < 2:
            continue
        for target_id in peer_ids:
            frontier_requirements = _frontier_requirements(
                target_id,
                conflicted_frontiers,
                structural_leaf_sets,
            )
            if frontier_requirements.conflicted:
                continue
            eligible = eligible_contributors(target_id)
            sealed = [item for item in eligible if item.accounted.sealed]
            if not sealed or len(sealed) == len(eligible):
                continue
            # Resolve complete sealed derivations, including sums assembled
            # from several contributors. Expose the recipe even if the target
            # already resolves identically: a complete copy has higher priority
            # for its peers than independently aggregating the same origins.
            resolution = _eligible_resolution(
                sealed,
                required_units(target_id),
                frontier_requirements.origins,
            )
            if resolution.records:
                peer_additions[target_id] = resolution.records[0]

    for target_id, record in peer_additions.items():
        states[target_id] = (*states[target_id], record)

    return recipe_source_ids


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
    coverage_cache: CoverageCache = {}
    contested_cache: dict[tuple[str, str], bool] = {}
    required_units_cache: dict[str, EntitySet] = {}
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
            required = required_units_cache.get(target_id)
            if required is None:
                required = _required_target_units(
                    target_id,
                    structural_leaf_sets,
                    origin_units,
                    shadowed_origin_ids,
                )
                required_units_cache[target_id] = required
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
                coverage_cache,
                contested_cache,
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
            merged_records = {
                _accounted_key(record): record for record in states[target_id]
            }
            merged_records.update(records_by_key)
            states[target_id] = tuple(
                merged_records[key] for key in sorted(merged_records)
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

    recipe_source_ids = _recipe_source_ids(
        states,
        dynamic_target_ids,
        conflicted_frontiers,
        structural_leaf_sets,
        origin_units,
        entities_by_id,
        inclusive_ancestors,
        inclusive_descendants,
        shadowed_origin_ids,
    )
    _discover_provenance_recipes(
        states,
        recipe_source_ids,
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
