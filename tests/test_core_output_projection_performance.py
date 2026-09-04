# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial synthetic regression cases for entity projection search."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import replace

import mlia.core.output_projection as projection
import mlia.core.output_schema as schema
from mlia.core.output_projection import project_entity_breakdowns
from tests.utils.entity_projection_scenarios import (
    breakdown,
    dense_overlapping_cover_result,
    many_alternative_cover_result,
    overlapping_recipe_groups_result,
    practical_rife_scale_result,
)
from tests.utils.entity_projection_scenarios import (
    result as projection_result,
)


def _breakdown_dicts_for(
    result: schema.Result,
    entity_id: str,
) -> list[dict]:
    """Serialize breakdowns for one entity in an already projected result."""
    return [item.to_dict() for item in result.breakdowns if item.entity_id == entity_id]


def _values(result: schema.Result) -> dict[str, int | float | None]:
    """Return the first cycles value materialized for each entity."""
    return {
        item.entity_id: item.metrics[0].value
        for item in result.breakdowns
        if item.metrics and item.metrics[0].name == "cycles"
    }


def test_dense_overlapping_graph_matches_semantically_minimal_graph() -> None:
    """Overlapping alternatives must not change a stronger exact projection."""
    minimal = project_entity_breakdowns(
        dense_overlapping_cover_result(leaf_count=7, max_span_width=0)
    )
    dense_input = dense_overlapping_cover_result(leaf_count=7, max_span_width=3)
    dense = project_entity_breakdowns(dense_input)
    reversed_dense = project_entity_breakdowns(
        replace(
            dense_input,
            entities=list(reversed(dense_input.entities)),
            breakdowns=list(reversed(dense_input.breakdowns)),
        )
    )

    expected = [
        {
            "entity_id": "target",
            "metrics": [
                {
                    "name": "cycles",
                    "unit": "cycles",
                    "value": 777,
                    "aggregation": "max",
                }
            ],
        }
    ]
    assert _breakdown_dicts_for(minimal, "target") == expected
    assert _breakdown_dicts_for(dense, "target") == expected
    assert _breakdown_dicts_for(reversed_dense, "target") == expected


def test_many_equivalent_alternatives_match_one_donor_per_leaf() -> None:
    """Interchangeable donors should collapse without altering SUM semantics."""
    leaf_count = 32
    minimal = project_entity_breakdowns(
        many_alternative_cover_result(
            leaf_count=leaf_count,
            alternatives_per_leaf=1,
            equivalent_alternatives=True,
        )
    )
    alternatives = project_entity_breakdowns(
        many_alternative_cover_result(
            leaf_count=leaf_count,
            alternatives_per_leaf=8,
            equivalent_alternatives=True,
        )
    )

    expected = [
        {
            "entity_id": "target",
            "metrics": [
                {
                    "name": "cycles",
                    "unit": "cycles",
                    "value": sum(range(1, leaf_count + 1)),
                    "aggregation": "sum",
                }
            ],
        }
    ]
    assert _breakdown_dicts_for(minimal, "target") == expected
    assert _breakdown_dicts_for(alternatives, "target") == expected
    assert alternatives.warnings == []


def test_overlapping_recipe_groups_converge_deterministically() -> None:
    """Many new recipes should survive later fixed-point passes and reordering."""
    group_count = 12
    forward_input = overlapping_recipe_groups_result(group_count)
    forward = project_entity_breakdowns(forward_input)
    backward = project_entity_breakdowns(
        replace(
            forward_input,
            entities=list(reversed(forward_input.entities)),
            breakdowns=list(reversed(forward_input.breakdowns)),
        )
    )

    expected = {
        entity_id: value
        for index in range(group_count)
        for entity_id, value in (
            (f"group-{index:02d}-bridge", 25 + 2 * index),
            (f"group-{index:02d}-target", 25 + 2 * index),
        )
    }
    forward_values = _values(forward)
    backward_values = _values(backward)

    assert {entity_id: forward_values[entity_id] for entity_id in expected} == expected
    assert {entity_id: backward_values[entity_id] for entity_id in expected} == expected
    assert backward_values == forward_values


def test_large_conflicting_cover_is_deterministic_and_target_local() -> None:
    """A pathological conflict must not suppress an independent safe projection."""
    result = many_alternative_cover_result(
        leaf_count=13,
        alternatives_per_leaf=2,
        equivalent_alternatives=False,
        include_independent_projection=True,
    )
    reversed_result = replace(
        result,
        entities=list(reversed(result.entities)),
        breakdowns=list(reversed(result.breakdowns)),
    )

    projected = project_entity_breakdowns(result)
    reversed_projected = project_entity_breakdowns(reversed_result)

    assert projected.breakdowns[: len(result.breakdowns)] == result.breakdowns
    assert _breakdown_dicts_for(projected, "target") == []
    assert _values(projected)["stable-target"] == 101
    assert projected.warnings == []
    assert reversed_projected.warnings == []
    assert _breakdown_dicts_for(reversed_projected, "target") == []
    assert _values(reversed_projected)["stable-target"] == 101


def test_two_interacting_conflicting_covers_complete_without_fallback() -> None:
    """Interacting targets must not make recipe-state merging quadratic."""
    result = many_alternative_cover_result(
        leaf_count=13,
        alternatives_per_leaf=2,
        equivalent_alternatives=False,
        target_ids=("target-a", "target-b"),
    )

    projected = project_entity_breakdowns(result)

    assert projected.breakdowns[: len(result.breakdowns)] == result.breakdowns
    assert _breakdown_dicts_for(projected, "target-a") == []
    assert _breakdown_dicts_for(projected, "target-b") == []
    assert projected.warnings == []


def test_recipe_producer_can_reach_consumer_through_represented_extras() -> None:
    """A complete recipe may feed a disjoint scope through origin extras."""
    entity = schema.Entity
    source_kind = schema.ENTITY_KIND_SOURCE_OPERATOR
    projected = project_entity_breakdowns(
        projection_result(
            [
                entity(id="P", kind="producer", name="P", child_ids=["x", "y"]),
                entity(id="T", kind="consumer", name="T", child_ids=["a", "b"]),
                entity(id="d1", kind="donor", name="d1", child_ids=["x", "a"]),
                entity(id="d2", kind="donor", name="d2", child_ids=["y", "b"]),
                entity(id="alt1", kind="donor", name="alt1", child_ids=["a", "z"]),
                entity(id="alt2", kind="donor", name="alt2", child_ids=["b", "w"]),
                *[
                    entity(id=entity_id, kind=source_kind, name=entity_id)
                    for entity_id in "xyabzw"
                ],
            ],
            [
                breakdown("d1", 2),
                breakdown("d2", 3),
                breakdown("alt1", 4),
                breakdown("alt2", 3),
            ],
            [
                schema.EntityKind(id="producer", child_kinds=[source_kind]),
                schema.EntityKind(id="consumer", child_kinds=[source_kind]),
                schema.EntityKind(id="donor", child_kinds=[source_kind]),
            ],
        )
    )

    assert _values(projected)["P"] == 12


def _private_figure_set(
    value: int | float,
    samples: int | None = None,
) -> projection.FigureSet:
    """Create one canonical private figure set for solver comparisons."""
    return (
        projection._FigureBreakdown(
            qualifiers_key="{}",
            qualifiers={},
            metrics=(
                schema.Metric(
                    name="cycles",
                    value=value,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                    samples=samples,
                ),
            ),
        ),
    )


def _private_contributor(
    origins: set[str],
    covered_units: set[str],
    value: int | float,
    *,
    extras: set[str] | None = None,
    samples: int | None = None,
    sealed: bool,
) -> projection._EligibleContributor:
    """Create one contributor for optimized/reference solver comparisons."""
    return projection._EligibleContributor(
        entity_ids=frozenset({"donor/" + "+".join(sorted(origins))}),
        accounted=projection._AccountedFigures(
            _private_figure_set(value, samples),
            frozenset(origins),
            sealed=sealed,
        ),
        covered_units=frozenset(covered_units),
        extras=frozenset(extras or set()),
    )


def _resolution_signature(
    derivations: Iterable[projection._Derivation],
    priority: int,
) -> tuple[bool, tuple[str, ...]]:
    """Return the observable conflict/figure outcome of one solver stream."""
    resolution = projection._resolve_priority_derivations(
        derivations,
        0 if priority == projection._PRIORITY_NO_EXTRAS else 1,
    )
    return (
        resolution.conflicted,
        tuple(record.figure_key for record in resolution.records),
    )


def _solver_signatures(
    contributors: list[projection._EligibleContributor],
    required_units: frozenset[str],
    priority: int,
    required_origins: frozenset[str],
) -> tuple[tuple[bool, tuple[str, ...]], tuple[bool, tuple[str, ...]]]:
    """Resolve one problem through final and reference solver behavior."""
    optimized = _resolution_signature(
        projection._iter_derivations_at_priority(
            contributors,
            required_units,
            priority,
            required_origins,
        ),
        priority,
    )
    reference = _resolution_signature(
        projection._iter_derivations_at_priority_reference(
            contributors,
            required_units,
            priority,
            required_origins,
        ),
        priority,
    )
    return optimized, reference


def test_final_solver_preserves_optional_origin_conflict() -> None:
    """A required-only cover must not hide an optional-origin competitor."""
    contributors = [
        _private_contributor({"a", "b"}, {"u1"}, 2, sealed=False),
        _private_contributor({"c", "d"}, {"u2"}, 7, sealed=False),
        _private_contributor(
            {"a", "b", "c"},
            {"u1", "u2"},
            5,
            sealed=False,
        ),
    ]

    optimized, reference = _solver_signatures(
        contributors,
        frozenset({"u1", "u2"}),
        projection._PRIORITY_NO_EXTRAS,
        frozenset({"a", "b", "c"}),
    )

    assert optimized == reference == (True, ())


def test_fractional_direct_atoms_match_reference_with_samples() -> None:
    """Exactly reconstructable fractional contributors may use the atom solver."""
    contributors = [
        _private_contributor({"a"}, {"u1"}, 0.25, samples=1, sealed=True),
        _private_contributor({"b"}, {"u2"}, 0.5, samples=2, sealed=True),
        _private_contributor(
            {"a", "b"},
            {"u1", "u2"},
            0.75,
            samples=3,
            sealed=False,
        ),
    ]
    required_units = frozenset({"u1", "u2"})
    required_origins = frozenset({"a", "b"})
    search = projection._derivation_search_index(
        contributors,
        required_units,
        projection._PRIORITY_NO_EXTRAS,
        required_origins,
    )

    assert search is not None
    assert (
        projection._direct_atom_derivations(
            search,
            projection._PRIORITY_NO_EXTRAS,
        )
        is not None
    )
    assert (
        _solver_signatures(
            contributors,
            required_units,
            projection._PRIORITY_NO_EXTRAS,
            required_origins,
        )
        == ((False, (contributors[2].accounted.figure_key,)),) * 2
    )


def test_inexact_fractional_composite_matches_reference_fallback() -> None:
    """A rounded composite must retain the reference solver's conflict semantics."""
    contributors = [
        _private_contributor({"a"}, {"u1"}, 0.1, sealed=True),
        _private_contributor({"b"}, {"u2"}, 0.2, sealed=True),
        _private_contributor(
            {"a", "b"},
            {"u1", "u2"},
            0.3,
            sealed=False,
        ),
    ]

    optimized, reference = _solver_signatures(
        contributors,
        frozenset({"u1", "u2"}),
        projection._PRIORITY_NO_EXTRAS,
        frozenset({"a", "b"}),
    )

    assert optimized == reference == (True, ())


def test_final_solver_matches_reference_on_random_small_cases() -> None:
    """Structural optimization must preserve final resolution semantics."""
    randomizer = random.Random(20260901)
    for case in range(250):
        origins = [f"o{index}" for index in range(randomizer.randint(2, 6))]
        units = [f"u{index}" for index in range(randomizer.randint(1, 5))]
        extras = [f"e{index}" for index in range(randomizer.randint(0, 2))]
        use_float = case % 3 == 0
        use_samples = case % 5 == 0
        values = {
            origin: (index + 1) / 10 if use_float else index + 1
            for index, origin in enumerate(origins)
        }
        coverage = {
            origin: {unit for unit in units if randomizer.random() < 0.45}
            for origin in origins
        }
        for unit in units:
            if not any(unit in coverage[origin] for origin in origins):
                coverage[randomizer.choice(origins)].add(unit)
        origin_extras = {
            origin: {extra for extra in extras if randomizer.random() < 0.3}
            for origin in origins
        }
        contributors = [
            _private_contributor(
                {origin},
                coverage[origin],
                values[origin],
                extras=origin_extras[origin],
                samples=1 if use_samples else None,
                sealed=bool(randomizer.getrandbits(1)),
            )
            for origin in origins
        ]
        seen = {frozenset({origin}) for origin in origins}
        for _unused in range(randomizer.randint(len(origins), 3 * len(origins))):
            selected = frozenset(
                randomizer.sample(
                    origins,
                    randomizer.randint(2, len(origins)),
                )
            )
            if selected in seen:
                continue
            seen.add(selected)
            ordered = sorted(selected)
            contributors.append(
                _private_contributor(
                    set(selected),
                    set().union(*(coverage[origin] for origin in ordered)),
                    sum(values[origin] for origin in ordered),
                    extras=set().union(*(origin_extras[origin] for origin in ordered)),
                    samples=len(selected) if use_samples else None,
                    sealed=bool(randomizer.getrandbits(1)),
                )
            )

        required_origins = frozenset(
            randomizer.sample(
                origins,
                randomizer.randint(1, len(origins)),
            )
        )
        required_units = frozenset(
            randomizer.sample(units, randomizer.randint(1, len(units)))
        )
        for priority in (
            projection._PRIORITY_NO_EXTRAS,
            projection._PRIORITY_WITH_EXTRAS,
        ):
            optimized, reference = _solver_signatures(
                contributors,
                required_units,
                priority,
                required_origins,
            )
            assert optimized == reference, (case, priority)


def test_minimum_extra_search_handles_globally_incompatible_zero_cost_options() -> None:
    """Origin overlap must not trigger a powerset search over possible extras."""
    option_count = 16
    contributors = [
        *[
            _private_contributor(
                {"shared"},
                {f"u{index}"},
                1,
                sealed=False,
            )
            for index in range(option_count)
        ],
        *[
            _private_contributor(
                {f"o{index}"},
                {f"u{index}"},
                1,
                extras={f"e{index}"},
                sealed=False,
            )
            for index in range(option_count)
        ],
    ]

    optimized, reference = _solver_signatures(
        contributors,
        frozenset(f"u{index}" for index in range(option_count)),
        projection._PRIORITY_WITH_EXTRAS,
        frozenset({"shared"}),
    )

    assert optimized == reference
    assert optimized[0] is False
    assert optimized[1]


def test_final_solver_handles_more_than_one_thousand_mandatory_origins() -> None:
    """Large mandatory origin sets must complete without Python recursion."""
    origin_count = 1100
    contributors = [
        _private_contributor(
            {f"o{index}"},
            {f"u{index}"},
            1,
            sealed=True,
        )
        for index in range(origin_count)
    ]

    derivations = list(
        projection._iter_derivations_at_priority(
            contributors,
            frozenset(f"u{index}" for index in range(origin_count)),
            projection._PRIORITY_NO_EXTRAS,
            frozenset(f"o{index}" for index in range(origin_count)),
        )
    )

    assert len(derivations) == 1
    assert derivations[0].accounted.figures[0].metrics[0].value == origin_count


def test_practical_rife_scale_supports_fractional_values_and_samples() -> None:
    """Realistic fractional metrics should stay on the direct-atom path."""
    result, expected_total = practical_rife_scale_result(use_floats_and_samples=True)

    projected = project_entity_breakdowns(result)
    for entity_id in ("nn_module/<root>", "nn_module/L__self__"):
        target = [item for item in projected.breakdowns if item.entity_id == entity_id]
        assert len(target) == 1
        assert target[0].metrics[0].value == expected_total
        assert target[0].metrics[0].samples == 333
