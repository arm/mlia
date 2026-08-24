# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Normalized entity-graph construction and validation.

Entity relationships may be declared from either direction: ``parent_ids`` on a
child and ``child_ids`` on a parent describe the same directed parent-to-child
edge. Reciprocal declarations are therefore optional. This module validates the
result-local identity space first, merges both declaration styles, rejects every
unresolved reference, and rejects directed cycles (including self-cycles).

The resulting :class:`EntityGraph` is a valid DAG. Consumers such as breakdown
projection can consequently resolve terminal descendants without defensive
cycle handling of their own.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityGraphDeclaration:
    """One result-local entity and its directed relationship declarations."""

    id: str  # pylint: disable=invalid-name
    parent_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    source_index: int = 0


@dataclass(frozen=True)
class EntityGraphIssue:
    """One graph validation problem with enough detail for schema diagnostics."""

    kind: str
    entity_id: str | None = None
    entity_index: int | None = None
    field_name: str | None = None
    reference_index: int | None = None
    reference: str | None = None
    cycle: tuple[str, ...] = ()

    def message(self, result_index: int | None = None) -> str:
        """Format the issue for a standalone result or standardized output."""
        if self.kind == "duplicate_id":
            suffix = (
                f" within result {result_index}" if result_index is not None else ""
            )
            return f"Entity id '{self.entity_id}' must be unique{suffix}"

        if self.kind == "unresolved_reference":
            path = (
                f"entities[{self.entity_index}].{self.field_name}"
                f"[{self.reference_index}]"
            )
            if result_index is not None:
                path = f"results[{result_index}].{path}"
                suffix = f" within result {result_index}"
            else:
                suffix = " within the result"
            return (
                f"Entity reference '{self.reference}' at {path} does not resolve"
                f"{suffix}"
            )

        if self.kind == "cycle":
            path = " -> ".join(repr(entity_id) for entity_id in self.cycle)
            location = f" in result {result_index}" if result_index is not None else ""
            return f"Entity graph{location} contains a directed cycle: {path}"

        return "Entity graph is invalid"


class EntityGraphValidationError(ValueError):
    """Raised when entity declarations do not form a result-local DAG."""

    def __init__(self, issues: list[EntityGraphIssue]) -> None:
        """Initialize the error with all detected graph issues."""
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message() for issue in issues))


@dataclass(frozen=True)
class EntityGraph:
    """A normalized, validated entity DAG."""

    children: dict[str, frozenset[str]]
    parents: dict[str, frozenset[str]]
    topological_order: tuple[str, ...]

    def structural_leaf_sets(self) -> dict[str, frozenset[str]]:
        """Return each entity's terminal descendants.

        A terminal entity is its own structural leaf. A non-terminal entity's
        leaves are the union of all terminal descendants reachable through the
        normalized child graph.
        """
        leaf_sets: dict[str, frozenset[str]] = {}
        for entity_id in reversed(self.topological_order):
            child_ids = self.children[entity_id]
            if not child_ids:
                leaf_sets[entity_id] = frozenset({entity_id})
            else:
                leaf_sets[entity_id] = frozenset(
                    leaf_id for child_id in child_ids for leaf_id in leaf_sets[child_id]
                )
        return leaf_sets

    def inclusive_ancestors(self) -> dict[str, frozenset[str]]:
        """Return each entity and every ancestor that can reach it."""
        ancestors: dict[str, frozenset[str]] = {}
        for entity_id in self.topological_order:
            ancestors[entity_id] = frozenset(
                {entity_id}
                | {
                    ancestor_id
                    for parent_id in self.parents[entity_id]
                    for ancestor_id in ancestors[parent_id]
                }
            )
        return ancestors


def _find_directed_cycle(
    children: dict[str, set[str]],
) -> tuple[str, ...] | None:
    """Find one deterministic directed cycle without recursive traversal."""
    state: dict[str, int] = {entity_id: 0 for entity_id in children}

    for root_id in sorted(children):
        if state[root_id] != 0:
            continue

        path = [root_id]
        positions = {root_id: 0}
        state[root_id] = 1
        stack: list[tuple[str, Iterator[str]]] = [
            (root_id, iter(sorted(children[root_id])))
        ]

        while stack:
            entity_id, child_iterator = stack[-1]
            try:
                child_id = next(child_iterator)
            except StopIteration:
                stack.pop()
                state[entity_id] = 2
                positions.pop(entity_id)
                path.pop()
                continue

            child_state = state[child_id]
            if child_state == 1:
                cycle_start = positions[child_id]
                return tuple([*path[cycle_start:], child_id])
            if child_state == 2:
                continue

            state[child_id] = 1
            positions[child_id] = len(path)
            path.append(child_id)
            stack.append((child_id, iter(sorted(children[child_id]))))

    return None


def validate_entity_graph(
    declarations: list[EntityGraphDeclaration],
) -> EntityGraph:
    """Validate and normalize result-local entity declarations.

    Raises:
        EntityGraphValidationError: if IDs are duplicated, a parent/child
            reference is unresolved, or the normalized graph contains a cycle.
    """
    issues: list[EntityGraphIssue] = []
    declarations_by_id: dict[str, list[EntityGraphDeclaration]] = {}
    for declaration in declarations:
        declarations_by_id.setdefault(declaration.id, []).append(declaration)

    for entity_id, matches in sorted(declarations_by_id.items()):
        if len(matches) > 1:
            issues.append(EntityGraphIssue("duplicate_id", entity_id=entity_id))

    known_ids = set(declarations_by_id)
    for declaration in declarations:
        for field_name, references in (
            ("parent_ids", declaration.parent_ids),
            ("child_ids", declaration.child_ids),
        ):
            for reference_index, reference in enumerate(references):
                if reference not in known_ids:
                    issues.append(
                        EntityGraphIssue(
                            "unresolved_reference",
                            entity_id=declaration.id,
                            entity_index=declaration.source_index,
                            field_name=field_name,
                            reference_index=reference_index,
                            reference=reference,
                        )
                    )

    if issues:
        raise EntityGraphValidationError(issues)

    children: dict[str, set[str]] = {entity_id: set() for entity_id in known_ids}
    parents: dict[str, set[str]] = {entity_id: set() for entity_id in known_ids}
    for declaration in declarations:
        for child_id in declaration.child_ids:
            children[declaration.id].add(child_id)
            parents[child_id].add(declaration.id)
        for parent_id in declaration.parent_ids:
            children[parent_id].add(declaration.id)
            parents[declaration.id].add(parent_id)

    cycle = _find_directed_cycle(children)
    if cycle is not None:
        raise EntityGraphValidationError([EntityGraphIssue("cycle", cycle=cycle)])

    remaining_parents = {
        entity_id: len(parent_ids) for entity_id, parent_ids in parents.items()
    }
    ready = [
        entity_id
        for entity_id, parent_count in remaining_parents.items()
        if parent_count == 0
    ]
    heapq.heapify(ready)
    topological_order: list[str] = []
    while ready:
        entity_id = heapq.heappop(ready)
        topological_order.append(entity_id)
        for child_id in sorted(children[entity_id]):
            remaining_parents[child_id] -= 1
            if remaining_parents[child_id] == 0:
                heapq.heappush(ready, child_id)

    return EntityGraph(
        children={
            entity_id: frozenset(child_ids) for entity_id, child_ids in children.items()
        },
        parents={
            entity_id: frozenset(parent_ids)
            for entity_id, parent_ids in parents.items()
        },
        topological_order=tuple(topological_order),
    )
