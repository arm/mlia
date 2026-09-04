# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic entity-projection scenarios shared by regression and benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import mlia.core.output_schema as schema


def metric(
    value: int | float,
    aggregation: schema.AggregationType = schema.AggregationType.SUM,
    samples: int | None = None,
) -> schema.Metric:
    """Create one synthetic cycles metric."""
    return schema.Metric(
        name="cycles",
        value=value,
        unit="cycles",
        aggregation=aggregation,
        samples=samples,
    )


def breakdown(
    entity_id: str,
    value: int | float,
    aggregation: schema.AggregationType = schema.AggregationType.SUM,
    samples: int | None = None,
) -> schema.Breakdown:
    """Create one synthetic authoritative breakdown."""
    return schema.Breakdown(
        entity_id=entity_id,
        metrics=[metric(value, aggregation, samples)],
    )


def result(
    entities: Iterable[schema.Entity],
    breakdowns: Iterable[schema.Breakdown],
    entity_kinds: Iterable[schema.EntityKind] = (),
) -> schema.Result:
    """Create one synthetic performance result."""
    return schema.Result(
        kind=schema.ResultKind.PERFORMANCE,
        status=schema.ResultStatus.OK,
        producer="entity-projection-test",
        entity_kinds=list(entity_kinds),
        entities=list(entities),
        breakdowns=list(breakdowns),
    )


def many_alternative_cover_result(
    *,
    leaf_count: int,
    alternatives_per_leaf: int,
    equivalent_alternatives: bool,
    include_independent_projection: bool = False,
    target_ids: tuple[str, ...] = ("target",),
) -> schema.Result:
    """Build targets with many interchangeable or conflicting choices per leaf."""
    leaf_ids = [f"leaf-{index:02d}" for index in range(leaf_count)]
    entities = [
        *[
            schema.Entity(
                id=target_id,
                kind=schema.ENTITY_KIND_CODE_STACK,
                name=target_id,
                child_ids=leaf_ids,
            )
            for target_id in target_ids
        ],
        *[
            schema.Entity(
                id=leaf_id,
                kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                name=leaf_id,
            )
            for leaf_id in leaf_ids
        ],
        *[
            schema.Entity(
                id=f"donor-{leaf_index:02d}-{alternative_index:02d}",
                kind=schema.ENTITY_KIND_CODE_STACK,
                name=f"donor-{leaf_index:02d}-{alternative_index:02d}",
                child_ids=[leaf_ids[leaf_index]],
            )
            for leaf_index in range(leaf_count)
            for alternative_index in range(alternatives_per_leaf)
        ],
    ]
    breakdowns = [
        *[schema.Breakdown(entity_id=leaf_id, metrics=[]) for leaf_id in leaf_ids],
        *[
            breakdown(
                f"donor-{leaf_index:02d}-{alternative_index:02d}",
                leaf_index + 1
                if equivalent_alternatives
                else leaf_index + alternative_index + 1,
            )
            for leaf_index in range(leaf_count)
            for alternative_index in range(alternatives_per_leaf)
        ],
    ]

    if include_independent_projection:
        entities.extend(
            [
                schema.Entity(
                    id="stable-target",
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name="stable target",
                    child_ids=["stable-leaf"],
                ),
                schema.Entity(
                    id="stable-donor",
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name="stable donor",
                    child_ids=["stable-leaf"],
                ),
                schema.Entity(
                    id="stable-leaf",
                    kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                    name="stable leaf",
                ),
            ]
        )
        breakdowns.extend(
            [
                schema.Breakdown(entity_id="stable-leaf", metrics=[]),
                breakdown(
                    "stable-donor",
                    101,
                    schema.AggregationType.MAX,
                ),
            ]
        )

    return result(entities, breakdowns)


def dense_overlapping_cover_result(
    *,
    leaf_count: int,
    max_span_width: int,
) -> schema.Result:
    """Build contiguous overlapping donors plus one authoritative exact donor."""
    leaf_ids = [f"leaf-{index:02d}" for index in range(leaf_count)]
    spans = [
        (start, width)
        for width in range(1, max_span_width + 1)
        for start in range(leaf_count - width + 1)
    ]
    entities = [
        schema.Entity(
            id="target",
            kind=schema.ENTITY_KIND_CODE_STACK,
            name="target",
            child_ids=leaf_ids,
        ),
        schema.Entity(
            id="exact",
            kind=schema.ENTITY_KIND_CODE_STACK,
            name="exact",
            child_ids=leaf_ids,
        ),
        *[
            schema.Entity(
                id=leaf_id,
                kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                name=leaf_id,
            )
            for leaf_id in leaf_ids
        ],
        *[
            schema.Entity(
                id=f"span-{start:02d}-{width:02d}",
                kind=schema.ENTITY_KIND_CODE_STACK,
                name=f"span-{start:02d}-{width:02d}",
                child_ids=leaf_ids[start : start + width],
            )
            for start, width in spans
        ],
    ]
    breakdowns = [
        *[schema.Breakdown(entity_id=leaf_id, metrics=[]) for leaf_id in leaf_ids],
        breakdown("exact", 777, schema.AggregationType.MAX),
        *[
            breakdown(
                f"span-{start:02d}-{width:02d}",
                sum(range(start + 1, start + width + 1)),
            )
            for start, width in spans
        ],
    ]
    return result(entities, breakdowns)


def overlapping_recipe_groups_result(group_count: int) -> schema.Result:
    """Build independent overlap groups that introduce recipes before convergence."""
    entities: list[schema.Entity] = []
    breakdowns: list[schema.Breakdown] = []
    for index in range(group_count):
        prefix = f"group-{index:02d}"
        leaf_a = f"{prefix}-leaf-a"
        leaf_b = f"{prefix}-leaf-b"
        leaf_c = f"{prefix}-leaf-c"
        broad = f"{prefix}-broad"
        c_donor = f"{prefix}-c-donor"
        entities.extend(
            [
                schema.Entity(
                    id=f"{prefix}-target",
                    kind="projection_target",
                    name=f"{prefix} target",
                    child_ids=[broad, leaf_c],
                ),
                schema.Entity(
                    id=broad,
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name=f"{prefix} broad",
                    child_ids=[leaf_a, leaf_b],
                ),
                schema.Entity(
                    id=f"{prefix}-bridge",
                    kind="projection_bridge",
                    name=f"{prefix} bridge",
                    child_ids=[leaf_b, leaf_c],
                ),
                schema.Entity(
                    id=c_donor,
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name=f"{prefix} c donor",
                    child_ids=[leaf_c],
                ),
                schema.Entity(
                    id=leaf_a,
                    kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                    name=leaf_a,
                ),
                schema.Entity(
                    id=leaf_b,
                    kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                    name=leaf_b,
                ),
                schema.Entity(
                    id=leaf_c,
                    kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                    name=leaf_c,
                ),
            ]
        )
        breakdowns.extend(
            [
                breakdown(broad, 9 + index),
                breakdown(c_donor, 16 + index),
                breakdown(leaf_b, 18 + index, schema.AggregationType.MAX),
                breakdown(leaf_a, 16 + index),
            ]
        )

    return result(
        entities,
        breakdowns,
        [
            schema.EntityKind(
                id="projection_target",
                child_kinds=[
                    schema.ENTITY_KIND_CODE_STACK,
                    schema.ENTITY_KIND_SOURCE_OPERATOR,
                ],
            ),
            schema.EntityKind(
                id="projection_bridge",
                child_kinds=[schema.ENTITY_KIND_SOURCE_OPERATOR],
            ),
        ],
    )


def practical_rife_scale_result(
    *,
    use_floats_and_samples: bool = False,
) -> tuple[schema.Result, int | float]:
    """Build a Practical-RIFE-scale mandatory and optional origin problem."""
    source_count = 861
    extra_count = 23
    paired_frontier_count = 95
    optional_origin_count = 142
    source_ids = [f"source-{index:03d}" for index in range(source_count)]
    extra_ids = [f"extra-{index:02d}" for index in range(extra_count)]
    entities: list[schema.Entity] = [
        schema.Entity(
            id="nn_module/<root>",
            kind="nn_module",
            name="<root>",
            child_ids=["nn_module/L__self__"],
        ),
        schema.Entity(
            id="nn_module/L__self__",
            kind="nn_module",
            name="L__self__",
            child_ids=source_ids,
        ),
        *[
            schema.Entity(
                id=entity_id,
                kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                name=entity_id,
            )
            for entity_id in [*source_ids, *extra_ids]
        ],
    ]
    breakdowns: list[schema.Breakdown] = []
    expected_total: int | float = 0.0 if use_floats_and_samples else 0

    def add_origin(entity_id: str, child_ids: list[str], value: int) -> None:
        nonlocal expected_total
        resolved_value: int | float = (
            float(value) + 0.25 if use_floats_and_samples else value
        )
        entities.append(
            schema.Entity(
                id=entity_id,
                kind="chain" if entity_id.startswith("chain/") else "code_stack",
                name=entity_id,
                child_ids=child_ids,
            )
        )
        breakdowns.append(
            breakdown(
                entity_id,
                resolved_value,
                samples=1 if use_floats_and_samples else None,
            )
        )
        expected_total += resolved_value

    for index in range(paired_frontier_count):
        add_origin(f"chain/{index:03d}/exact", [source_ids[index]], index + 1)
        add_origin(
            f"chain/{index:03d}/residual",
            [source_ids[index], extra_ids[index % extra_count]],
            1000 + index,
        )
    add_origin(
        "chain/095/residual",
        [source_ids[paired_frontier_count], extra_ids[0]],
        2000,
    )

    optional_start = paired_frontier_count + 1
    optional_units = source_count - optional_start
    for index in range(optional_origin_count):
        start = optional_start + index * optional_units // optional_origin_count
        end = optional_start + (index + 1) * optional_units // optional_origin_count
        add_origin(
            f"code_stack/optional-{index:03d}",
            source_ids[start:end],
            3000 + index,
        )

    return (
        result(
            entities,
            breakdowns,
            [
                schema.EntityKind(
                    id="chain",
                    child_kinds=[schema.ENTITY_KIND_SOURCE_OPERATOR],
                ),
                schema.EntityKind(
                    id="nn_module",
                    parent_kinds=["nn_module"],
                    child_kinds=[
                        "nn_module",
                        schema.ENTITY_KIND_SOURCE_OPERATOR,
                    ],
                ),
            ],
        ),
        expected_total,
    )
