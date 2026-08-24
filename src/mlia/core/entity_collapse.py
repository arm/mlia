# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Generic entity collapse for standardized MLIA results."""

from __future__ import annotations

import logging
from dataclasses import replace
from fnmatch import fnmatchcase

import mlia.core.output_schema as schema
from mlia.core.entity_graph import (
    EntityGraph,
    EntityGraphDeclaration,
    validate_entity_graph,
)
from mlia.core.output_validation import (
    SchemaValidationError,
    validate_result_entity_kind_relationships,
)
from mlia.core.settings import CollapseRule

logger = logging.getLogger(__name__)


def _graph(result: schema.Result) -> EntityGraph:
    return validate_entity_graph(
        [
            EntityGraphDeclaration(
                entity.id,
                tuple(entity.parent_ids),
                tuple(entity.child_ids),
                index,
            )
            for index, entity in enumerate(result.entities)
        ]
    )


def _matches(entity: schema.Entity, rules: tuple[CollapseRule, ...]) -> bool:
    for rule in rules:
        if entity.kind != rule.kind:
            continue
        value = entity.attributes.get(rule.attribute)
        if isinstance(value, str) and any(
            fnmatchcase(value, pattern) for pattern in rule.globs
        ):
            return True
    return False


def _nearest_retained(
    starts: frozenset[str],
    relationships: dict[str, frozenset[str]],
    collapsed_ids: set[str],
    origin_id: str,
) -> list[str]:
    found: list[str] = []
    pending = sorted(starts)
    visited: set[str] = set()
    while pending:
        entity_id = pending.pop(0)
        if entity_id in visited:
            continue
        visited.add(entity_id)
        if entity_id not in collapsed_ids:
            if entity_id != origin_id:
                found.append(entity_id)
            continue
        pending.extend(sorted(relationships[entity_id]))
    return list(dict.fromkeys(found))


def _contracted_entities(
    result: schema.Result, graph: EntityGraph, collapsed_ids: set[str]
) -> list[schema.Entity]:
    """Return retained entities with relationships contracted through removals."""
    return [
        replace(
            entity,
            parent_ids=_nearest_retained(
                graph.parents[entity.id], graph.parents, collapsed_ids, entity.id
            ),
            child_ids=_nearest_retained(
                graph.children[entity.id], graph.children, collapsed_ids, entity.id
            ),
        )
        for entity in result.entities
        if entity.id not in collapsed_ids
    ]


def _safe_collapsed_ids(
    result: schema.Result, graph: EntityGraph, candidates: set[str]
) -> set[str]:
    """Return candidates whose cumulative contraction preserves kind semantics."""
    collapsed_ids: set[str] = set()
    for entity_id in sorted(candidates):
        trial_ids = {*collapsed_ids, entity_id}
        trial = replace(
            result,
            entities=_contracted_entities(result, graph, trial_ids),
        )
        trial_graph = _graph(trial)
        try:
            validate_result_entity_kind_relationships(trial, trial_graph)
        except SchemaValidationError:
            logger.warning(
                "Retaining entity '%s' because collapsing it would create an "
                "undeclared entity-kind relationship.",
                entity_id,
            )
        else:
            collapsed_ids = trial_ids
    return collapsed_ids


def collapse_entities(
    result: schema.Result, rules: tuple[CollapseRule, ...]
) -> schema.Result:
    """Remove matching entities and contract their normalized graph relationships.

    Matching is deliberately generic: exact entity kind, one named string
    attribute, and case-sensitive raw-value glob matching. The input is not
    mutated. Records targeting removed entities are discarded and removed IDs
    are pruned from advice.
    """
    graph = _graph(result)
    if not rules:
        return result

    candidates = {entity.id for entity in result.entities if _matches(entity, rules)}
    collapsed_ids = _safe_collapsed_ids(result, graph, candidates)
    if not collapsed_ids:
        return result

    entities = _contracted_entities(result, graph, collapsed_ids)
    discarded_breakdowns = [
        item for item in result.breakdowns if item.entity_id in collapsed_ids
    ]
    if discarded_breakdowns:
        logger.warning(
            "Discarding %d authoritative breakdown(s) targeting collapsed entities %s.",
            len(discarded_breakdowns),
            sorted({item.entity_id for item in discarded_breakdowns}),
        )

    advice: list[schema.Advice] = []
    for item in result.advice:
        affected = [
            entity_id
            for entity_id in item.affected_entity_ids
            if entity_id not in collapsed_ids
        ]
        if item.affected_entity_ids and not affected:
            continue
        advice.append(replace(item, affected_entity_ids=affected))

    collapsed = replace(
        result,
        entities=entities,
        breakdowns=[
            item for item in result.breakdowns if item.entity_id not in collapsed_ids
        ],
        checks=[item for item in result.checks if item.entity_id not in collapsed_ids],
        advice=advice,
    )
    collapsed_graph = _graph(collapsed)
    validate_result_entity_kind_relationships(collapsed, collapsed_graph)
    return collapsed
