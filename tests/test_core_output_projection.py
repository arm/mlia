# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standardized-output entity breakdown projection."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

import mlia.core.output_schema as schema
from mlia.core.output_projection import project_entity_breakdowns
from mlia.core.output_validation import SchemaValidationError


def _metric(
    value: float | int | None,
    *,
    name: str = "cycles",
    unit: str = "cycles",
    qualifiers: dict | None = None,
    aggregation: str | None = None,
    samples: int | None = None,
    reason: str = "not reported",
) -> schema.Metric:
    if value is None:
        return schema.Metric(
            name=name,
            value=None,
            unit=unit,
            qualifiers=qualifiers or {},
            availability=schema.MetricAvailability.UNAVAILABLE,
            reason=reason,
        )
    return schema.Metric(
        name=name,
        value=value,
        unit=unit,
        qualifiers=qualifiers or {},
        aggregation=aggregation,
        samples=samples,
    )


def _breakdown(
    entity_id: str,
    *metrics: schema.Metric,
    qualifiers: dict | None = None,
    breakdown_id: str | None = None,
) -> schema.Breakdown:
    return schema.Breakdown(
        entity_id=entity_id,
        metrics=list(metrics),
        qualifiers=qualifiers or {},
        id=breakdown_id,
    )


def _result(
    entities: Iterable[schema.Entity],
    breakdowns: Iterable[schema.Breakdown],
    entity_kinds: Iterable[schema.EntityKind] | None = None,
) -> schema.Result:
    entity_list = list(entities)
    if entity_kinds is None:
        entities_by_id = {entity.id: entity for entity in entity_list}
        child_kinds_by_parent: dict[str, set[str]] = {}
        custom_kinds = {
            entity.kind
            for entity in entity_list
            if entity.kind not in schema.WELL_KNOWN_ENTITY_KINDS
        }
        for entity in entity_list:
            for child_id in entity.child_ids:
                child = entities_by_id.get(child_id)
                if child is not None and entity.kind in custom_kinds:
                    child_kinds_by_parent.setdefault(entity.kind, set()).add(child.kind)
            for parent_id in entity.parent_ids:
                parent = entities_by_id.get(parent_id)
                if parent is not None and parent.kind in custom_kinds:
                    child_kinds_by_parent.setdefault(parent.kind, set()).add(
                        entity.kind
                    )
        resolved_entity_kinds = [
            schema.EntityKind(
                id=kind,
                child_kinds=sorted(child_kinds_by_parent.get(kind, set())),
            )
            for kind in sorted(custom_kinds)
        ]
    else:
        resolved_entity_kinds = list(entity_kinds)
    return schema.Result(
        kind=schema.ResultKind.PERFORMANCE,
        status=schema.ResultStatus.OK,
        producer="test",
        entity_kinds=resolved_entity_kinds,
        entities=entity_list,
        breakdowns=list(breakdowns),
    )


_CHAIN_SOURCE_ENTITY_KINDS = [
    schema.EntityKind(id="chain", child_kinds=["source_operator"]),
]


def _projected_for(result: schema.Result, entity_id: str) -> list[schema.Breakdown]:
    return [
        breakdown
        for breakdown in project_entity_breakdowns(result).breakdowns
        if breakdown.entity_id == entity_id
    ]


def test_projection_search_completes_without_a_defensive_fallback() -> None:
    entities = [
        schema.Entity(id="donor", kind="stats", name="donor", child_ids=["leaf"]),
        schema.Entity(id="target", kind="view", name="target", child_ids=["leaf"]),
        schema.Entity(
            id="leaf",
            kind="operator",
            name="leaf",
            parent_ids=["donor", "target"],
        ),
    ]
    result = _result(entities, [_breakdown("donor", _metric(7))])

    projected = project_entity_breakdowns(result)

    assert projected.breakdowns[: len(result.breakdowns)] == result.breakdowns
    assert [item for item in projected.breakdowns if item.entity_id == "target"] == [
        _breakdown("target", _metric(7))
    ]
    assert projected.warnings == result.warnings


@pytest.mark.parametrize("relationship_declaration", ["parents", "children", "both"])
def test_exact_leaf_set_projection_normalizes_relationships(
    relationship_declaration: str,
) -> None:
    """Equivalent leaf sets should copy figures for all relationship spellings."""
    parent_ids = ["donor", "target"] if relationship_declaration != "children" else []
    child_ids = ["leaf"] if relationship_declaration != "parents" else []
    entities = [
        schema.Entity(id="donor", kind="stats", name="donor", child_ids=child_ids),
        schema.Entity(id="target", kind="view", name="target", child_ids=child_ids),
        schema.Entity(
            id="leaf",
            kind="operator",
            name="leaf",
            parent_ids=parent_ids,
        ),
    ]
    donor_metric = _metric(
        7,
        qualifiers={"phase": "execute"},
        aggregation="sum",
        samples=3,
    )
    result = _result(
        entities,
        [
            _breakdown(
                "donor",
                donor_metric,
                qualifiers={"scenario": "warm"},
                breakdown_id="donor-breakdown",
            )
        ],
    )

    projected = _projected_for(result, "target")

    assert projected == [
        _breakdown(
            "target",
            donor_metric,
            qualifiers={"scenario": "warm"},
        )
    ]


def test_aggregation_uses_disjoint_complete_cover() -> None:
    """Numeric metrics should sum over a disjoint complete leaf cover."""
    result = _result(
        [
            schema.Entity(id="root", kind="group", name="root", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a", parent_ids=["root"]),
            schema.Entity(id="b", kind="operator", name="b", parent_ids=["root"]),
        ],
        [
            _breakdown(
                "a",
                _metric(2, aggregation="sum", samples=2),
                qualifiers={"run": "one"},
            ),
            _breakdown(
                "b",
                _metric(3, aggregation="sum", samples=4),
                qualifiers={"run": "one"},
            ),
        ],
    )

    assert _projected_for(result, "root") == [
        _breakdown(
            "root",
            _metric(5, aggregation="sum", samples=6),
            qualifiers={"run": "one"},
        )
    ]


@pytest.mark.parametrize("aggregation", ["mean", "max", "custom"])
def test_non_additive_aggregation_metadata_is_not_summed(aggregation: str) -> None:
    """Matching non-additive metadata must not be reinterpreted as a sum."""
    result = _result(
        [
            schema.Entity(id="root", kind="group", name="root", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown("a", _metric(2, aggregation=aggregation, samples=2)),
            _breakdown("b", _metric(3, aggregation=aggregation, samples=4)),
        ],
    )

    assert _projected_for(result, "root") == []


def test_implicit_additive_metrics_with_samples_are_not_summed() -> None:
    """Sample metadata requires an explicit additive aggregation mode."""
    result = _result(
        [
            schema.Entity(id="root", kind="group", name="root", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown("a", _metric(2, samples=2)),
            _breakdown("b", _metric(3, samples=4)),
        ],
    )

    assert _projected_for(result, "root") == []


def test_absent_leaf_is_inferred_to_have_no_attributable_statistics() -> None:
    """An unaccounted terminal leaf may remain uncovered by a complete projection."""
    result = _result(
        [
            schema.Entity(
                id="root", kind="group", name="root", child_ids=["a", "b", "c"]
            ),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
            schema.Entity(id="c", kind="operator", name="c"),
        ],
        [
            _breakdown("a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("b", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
    )

    assert _projected_for(result, "root") == [
        _breakdown("root", _metric(5, aggregation=schema.AggregationType.SUM))
    ]


def test_distinct_sum_origins_can_overlap_graph_coverage() -> None:
    """Distinct authoritative origins remain additive despite shared coverage."""
    result = _result(
        [
            schema.Entity(
                id="root", kind="group", name="root", child_ids=["a", "b", "c"]
            ),
            schema.Entity(id="left", kind="group", name="left", child_ids=["a", "b"]),
            schema.Entity(id="right", kind="group", name="right", child_ids=["b", "c"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
            schema.Entity(id="c", kind="operator", name="c"),
        ],
        [
            _breakdown("left", _metric(5, aggregation=schema.AggregationType.SUM)),
            _breakdown("right", _metric(7, aggregation=schema.AggregationType.SUM)),
        ],
    )

    # Two chains may cover different scheduled portions of the same source
    # operator. Their shared graph coverage does not make their separately owned
    # authoritative SUM measurements duplicates.
    assert _projected_for(result, "root") == [
        _breakdown("root", _metric(12, aggregation=schema.AggregationType.SUM))
    ]


def test_duplicate_identical_leaf_sets_are_alternatives() -> None:
    """Equivalent donors should be copied once rather than added together."""
    result = _result(
        [
            schema.Entity(
                id="first", kind="code_stack", name="first", child_ids=["leaf"]
            ),
            schema.Entity(
                id="second", kind="code_stack", name="second", child_ids=["leaf"]
            ),
            schema.Entity(
                id="target", kind="code_stack", name="target", child_ids=["leaf"]
            ),
            schema.Entity(id="leaf", kind="source_operator", name="leaf"),
        ],
        [_breakdown("first", _metric(5)), _breakdown("second", _metric(5))],
    )

    assert _projected_for(result, "target") == [_breakdown("target", _metric(5))]


def test_many_equivalent_donors_per_leaf_collapse_before_cover_search() -> None:
    """Large interchangeable alternatives should produce one semantic SUM cover."""
    leaf_count = 24
    alternatives_per_leaf = 3
    leaf_ids = [f"leaf-{index}" for index in range(leaf_count)]
    entities = [
        schema.Entity(id="target", kind="target", name="target", child_ids=leaf_ids),
        *[
            schema.Entity(id=leaf_id, kind="source_operator", name=leaf_id)
            for leaf_id in leaf_ids
        ],
        *[
            schema.Entity(
                id=f"donor-{leaf_index}-{alternative_index}",
                kind="code_stack",
                name=f"donor-{leaf_index}-{alternative_index}",
                child_ids=[leaf_ids[leaf_index]],
            )
            for leaf_index in range(leaf_count)
            for alternative_index in range(alternatives_per_leaf)
        ],
    ]
    breakdowns = [
        _breakdown(
            f"donor-{leaf_index}-{alternative_index}",
            _metric(leaf_index + 1, aggregation=schema.AggregationType.SUM),
        )
        for leaf_index in range(leaf_count)
        for alternative_index in range(alternatives_per_leaf)
    ]

    assert _projected_for(_result(entities, breakdowns), "target") == [
        _breakdown(
            "target",
            _metric(
                sum(range(1, leaf_count + 1)),
                aggregation=schema.AggregationType.SUM,
            ),
        )
    ]


def test_conflicting_same_leaf_set_donors_are_skipped() -> None:
    """Conflicting alternatives should make their shared leaf set unresolved."""
    result = _result(
        [
            schema.Entity(
                id="first", kind="code_stack", name="first", child_ids=["leaf"]
            ),
            schema.Entity(
                id="second", kind="code_stack", name="second", child_ids=["leaf"]
            ),
            schema.Entity(
                id="target", kind="code_stack", name="target", child_ids=["leaf"]
            ),
            schema.Entity(id="leaf", kind="source_operator", name="leaf"),
        ],
        [_breakdown("first", _metric(5)), _breakdown("second", _metric(6))],
    )

    assert _projected_for(result, "target") == []


def test_direct_authoritative_parents_are_summed_for_source_and_code_stack() -> None:
    """Distinct direct parents are additive contributions on the nearest frontier."""
    result = _result(
        [
            schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["source"]),
            schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["source"]),
            schema.Entity(id="source", kind="source_operator", name="source"),
            schema.Entity(
                id="frame", kind="code_stack", name="frame", child_ids=["source"]
            ),
        ],
        [
            _breakdown("chain-a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    projected = project_entity_breakdowns(result)

    assert [
        breakdown
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "source"
    ] == [_breakdown("source", _metric(5, aggregation=schema.AggregationType.SUM))]
    assert [
        breakdown
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "frame"
    ] == [_breakdown("frame", _metric(5, aggregation=schema.AggregationType.SUM))]


def test_equal_direct_authoritative_parent_figures_are_distinct_contributions() -> None:
    """Equal figures from different direct parents must be added rather than deduped."""
    result = _result(
        [
            schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["source"]),
            schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["source"]),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [
            _breakdown("chain-a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(2, aggregation=schema.AggregationType.SUM)),
        ],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    assert _projected_for(result, "source") == [
        _breakdown("source", _metric(4, aggregation=schema.AggregationType.SUM))
    ]


def test_projection_rejects_undeclared_kind_relationship() -> None:
    """Projection should reject graph edges without declared kind semantics."""
    result = _result(
        [
            schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["source"]),
            schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["source"]),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [
            _breakdown("chain-a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
        [],
    )

    with pytest.raises(SchemaValidationError, match="not covered"):
        project_entity_breakdowns(result)


def test_mixed_direct_parent_blocks_exact_sibling_and_downstream_view() -> None:
    """A contested mixed parent must block partial attribution from an exact parent."""
    result = _result(
        [
            schema.Entity(
                id="exact-chain", kind="chain", name="exact", child_ids=["source"]
            ),
            schema.Entity(
                id="mixed-chain",
                kind="chain",
                name="mixed",
                child_ids=["source", "other"],
            ),
            schema.Entity(id="source", kind="source_operator", name="source"),
            schema.Entity(id="other", kind="source_operator", name="other"),
            schema.Entity(
                id="frame", kind="code_stack", name="frame", child_ids=["source"]
            ),
        ],
        [
            _breakdown(
                "exact-chain", _metric(2, aggregation=schema.AggregationType.SUM)
            ),
            _breakdown(
                "mixed-chain", _metric(10, aggregation=schema.AggregationType.SUM)
            ),
        ],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    projected = project_entity_breakdowns(result)

    assert not any(
        breakdown.entity_id in {"source", "frame"} for breakdown in projected.breakdowns
    )


def test_exact_multi_source_stack_copies_chain_without_disaggregation() -> None:
    """A stack matching a whole chain can copy its figures without splitting them."""
    metric = _metric(10, aggregation=schema.AggregationType.SUM)
    result = _result(
        [
            schema.Entity(
                id="chain",
                kind="chain",
                name="chain",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(
                id="frame",
                kind="code_stack",
                name="frame",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [_breakdown("chain", metric)],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    projected = project_entity_breakdowns(result)

    assert not any(
        breakdown.entity_id in {"source-a", "source-b"}
        for breakdown in projected.breakdowns
    )
    assert [
        breakdown
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "frame"
    ] == [_breakdown("frame", metric)]


def test_equal_coverage_view_accounts_for_every_overlapping_chain_origin() -> None:
    """An equal view must not copy one origin while omitting another full origin."""
    result = _result(
        [
            schema.Entity(
                id="chain-a",
                kind="chain",
                name="a",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(
                id="chain-b",
                kind="chain",
                name="b",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(
                id="frame",
                kind="code_stack",
                name="frame",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [
            _breakdown("chain-a", _metric(4, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(6, aggregation=schema.AggregationType.SUM)),
        ],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    projected = project_entity_breakdowns(result)

    assert not any(
        breakdown.entity_id in {"source-a", "source-b"}
        for breakdown in projected.breakdowns
    )
    assert [
        breakdown
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "frame"
    ] == [_breakdown("frame", _metric(10, aggregation=schema.AggregationType.SUM))]


def test_broader_view_combines_complete_multi_source_chain_origins() -> None:
    """A broader view may sum whole conflicted frontiers without disaggregation."""
    result = _result(
        [
            schema.Entity(
                id="chain-a",
                kind="chain",
                name="a",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(
                id="chain-b",
                kind="chain",
                name="b",
                child_ids=["source-c", "source-d"],
            ),
            schema.Entity(
                id="frame",
                kind="code_stack",
                name="frame",
                child_ids=["source-a", "source-b", "source-c", "source-d"],
            ),
            *[
                schema.Entity(
                    id=f"source-{suffix}",
                    kind="source_operator",
                    name=suffix,
                )
                for suffix in ("a", "b", "c", "d")
            ],
        ],
        [
            _breakdown("chain-a", _metric(4, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(6, aggregation=schema.AggregationType.SUM)),
        ],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    projected = project_entity_breakdowns(result)

    assert not any(
        breakdown.entity_id.startswith("source-") for breakdown in projected.breakdowns
    )
    assert [
        breakdown
        for breakdown in projected.breakdowns
        if breakdown.entity_id == "frame"
    ] == [_breakdown("frame", _metric(10, aggregation=schema.AggregationType.SUM))]


def test_direct_authoritative_parent_outranks_aggregate_ancestor() -> None:
    """A chain frontier is used without also counting its authoritative cascade."""
    result = _result(
        [
            schema.Entity(
                id="cascade", kind="cascade", name="cascade", child_ids=["chain"]
            ),
            schema.Entity(id="chain", kind="chain", name="chain", child_ids=["source"]),
            schema.Entity(id="source", kind="source_operator", name="source"),
            schema.Entity(
                id="frame", kind="code_stack", name="frame", child_ids=["source"]
            ),
        ],
        [
            _breakdown("cascade", _metric(10, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain", _metric(4, aggregation=schema.AggregationType.SUM)),
        ],
        [
            schema.EntityKind(id="cascade", child_kinds=["chain"]),
            *_CHAIN_SOURCE_ENTITY_KINDS,
        ],
    )

    assert _projected_for(result, "source") == [
        _breakdown("source", _metric(4, aggregation=schema.AggregationType.SUM))
    ]
    assert _projected_for(result, "frame") == [
        _breakdown("frame", _metric(4, aggregation=schema.AggregationType.SUM))
    ]


def test_shadowed_identical_aggregate_does_not_block_equivalent_view() -> None:
    """A serialized aggregate copy must not become an impossible required origin."""
    metric = _metric(30, aggregation=schema.AggregationType.SUM)
    result = _result(
        [
            schema.Entity(
                id="segment",
                kind="segment",
                name="segment",
                child_ids=["chain", "neutral"],
            ),
            schema.Entity(
                id="chain",
                kind="chain",
                name="chain",
                parent_ids=["segment"],
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(
                id="namespace",
                kind="namespace",
                name="namespace",
                child_ids=["source-a", "source-b", "neutral"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
            schema.Entity(id="neutral", kind="source_operator", name="neutral"),
        ],
        [_breakdown("chain", metric), _breakdown("segment", metric)],
        [
            schema.EntityKind(id="segment", child_kinds=["chain", "source_operator"]),
            schema.EntityKind(
                id="chain",
                parent_kinds=["segment"],
                child_kinds=["source_operator"],
            ),
            schema.EntityKind(id="namespace", child_kinds=["source_operator"]),
        ],
    )

    assert _projected_for(result, "namespace") == [_breakdown("namespace", metric)]


def test_incompatible_direct_authoritative_parents_suppress_projection() -> None:
    """Multiple frontier contributions require compatible explicit SUM metrics."""
    result = _result(
        [
            schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["source"]),
            schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["source"]),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [
            _breakdown("chain-a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(3, aggregation=schema.AggregationType.MAX)),
        ],
        _CHAIN_SOURCE_ENTITY_KINDS,
    )

    assert _projected_for(result, "source") == []


def test_resolved_frontiers_aggregate_into_parent_stack_and_module() -> None:
    """Downstream parents combine sealed chain origins without raw alternatives."""
    result = _result(
        [
            schema.Entity(
                id="conv-chain", kind="chain", name="conv", child_ids=["conv"]
            ),
            schema.Entity(
                id="resize-chain-a",
                kind="chain",
                name="resize-a",
                child_ids=["resize"],
            ),
            schema.Entity(
                id="resize-chain-b",
                kind="chain",
                name="resize-b",
                child_ids=["resize"],
            ),
            schema.Entity(id="conv", kind="source_operator", name="conv"),
            schema.Entity(id="resize", kind="source_operator", name="resize"),
            schema.Entity(
                id="conv-frame", kind="code_stack", name="conv", child_ids=["conv"]
            ),
            schema.Entity(
                id="resize-frame",
                kind="code_stack",
                name="resize",
                child_ids=["resize"],
            ),
            schema.Entity(
                id="stage-frame",
                kind="code_stack",
                name="stage",
                child_ids=["conv-frame", "resize-frame"],
            ),
            schema.Entity(
                id="stage-module",
                kind="nn_module",
                name="friendly_stage",
                child_ids=["conv", "resize"],
            ),
        ],
        [
            _breakdown(
                "conv-chain", _metric(10, aggregation=schema.AggregationType.SUM)
            ),
            _breakdown(
                "resize-chain-a",
                _metric(2, aggregation=schema.AggregationType.SUM),
            ),
            _breakdown(
                "resize-chain-b",
                _metric(3, aggregation=schema.AggregationType.SUM),
            ),
        ],
        [
            *_CHAIN_SOURCE_ENTITY_KINDS,
            schema.EntityKind(id="nn_module", child_kinds=["source_operator"]),
        ],
    )

    projected = project_entity_breakdowns(result)

    assert [item for item in projected.breakdowns if item.entity_id == "resize"] == [
        _breakdown("resize", _metric(5, aggregation=schema.AggregationType.SUM))
    ]
    for entity_id in ("stage-frame", "stage-module"):
        assert [
            item for item in projected.breakdowns if item.entity_id == entity_id
        ] == [
            _breakdown(entity_id, _metric(15, aggregation=schema.AggregationType.SUM))
        ]


def test_recursive_module_hierarchy_requires_complete_descendant_coverage() -> None:
    """A module branch is complete only after every measured descendant is covered."""
    entities = [
        schema.Entity(
            id="module-root",
            kind="nn_module",
            name="<root>",
            child_ids=["module-self"],
        ),
        schema.Entity(
            id="module-self",
            kind="nn_module",
            name="L__self__",
            child_ids=["high-stage", "medium-stage", "output-stage"],
        ),
        schema.Entity(
            id="high-stage",
            kind="nn_module",
            name="high",
            child_ids=["high-conv-module", "high-resize"],
        ),
        schema.Entity(
            id="high-conv-module",
            kind="nn_module",
            name="high-conv",
            child_ids=["high-conv"],
        ),
        schema.Entity(
            id="medium-stage",
            kind="nn_module",
            name="medium",
            child_ids=["medium-conv-module", "medium-resize"],
        ),
        schema.Entity(
            id="medium-conv-module",
            kind="nn_module",
            name="medium-conv",
            child_ids=["medium-conv"],
        ),
        schema.Entity(
            id="output-stage",
            kind="nn_module",
            name="output",
            child_ids=["output-conv"],
        ),
        schema.Entity(id="high-conv", kind="source_operator", name="high-conv"),
        schema.Entity(id="high-resize", kind="source_operator", name="high-resize"),
        schema.Entity(id="medium-conv", kind="source_operator", name="medium-conv"),
        schema.Entity(id="medium-resize", kind="source_operator", name="medium-resize"),
        schema.Entity(id="output-conv", kind="source_operator", name="output-conv"),
    ]
    chain_values = {
        "high-conv": [14593],
        "high-resize": [1148, 1148, 1148, 6656, 3697],
        "medium-conv": [5256],
        "medium-resize": [380, 380, 514, 2097, 1338],
        "output-conv": [2872],
    }
    breakdowns: list[schema.Breakdown] = []
    for source_id, values in chain_values.items():
        for index, value in enumerate(values):
            chain_id = f"chain-{source_id}-{index}"
            entities.append(
                schema.Entity(
                    id=chain_id,
                    kind="chain",
                    name=chain_id,
                    child_ids=[source_id],
                )
            )
            breakdowns.append(
                _breakdown(
                    chain_id,
                    _metric(value, aggregation=schema.AggregationType.SUM),
                )
            )

    result = _result(
        entities,
        breakdowns,
        [
            schema.EntityKind(id="chain", child_kinds=["source_operator"]),
            schema.EntityKind(
                id="nn_module",
                parent_kinds=["nn_module"],
                child_kinds=["nn_module", "source_operator"],
            ),
        ],
    )
    projected = project_entity_breakdowns(result)
    totals = {
        breakdown.entity_id: breakdown.metrics[0].value
        for breakdown in projected.breakdowns
        if breakdown.metrics and breakdown.metrics[0].name == "cycles"
    }

    assert totals["high-stage"] == 28390
    assert totals["medium-stage"] == 9965
    assert totals["output-stage"] == 2872
    assert totals["module-self"] == 41227
    assert totals["module-root"] == 41227


def test_projection_does_not_disaggregate_multi_leaf_donor() -> None:
    """A multi-leaf figure should not be allocated to its individual leaves."""
    result = _result(
        [
            schema.Entity(id="group", kind="group", name="group", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [_breakdown("group", _metric(5))],
    )

    projected = project_entity_breakdowns(result)

    assert [item.entity_id for item in projected.breakdowns] == ["group"]


def test_projection_propagates_across_independent_hierarchy_branches() -> None:
    """Stats leaves should project through source, code-stack, and module views."""
    result = _result(
        [
            schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["op-a"]),
            schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["op-b"]),
            schema.Entity(id="op-a", kind="source_operator", name="a"),
            schema.Entity(id="op-b", kind="source_operator", name="b"),
            schema.Entity(
                id="frame", kind="code_stack", name="frame", child_ids=["op-a"]
            ),
            schema.Entity(
                id="module",
                kind="nn_module",
                name="module",
                child_ids=["op-a", "op-b"],
            ),
        ],
        [
            _breakdown("chain-a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
        [
            schema.EntityKind(id="chain", child_kinds=["source_operator"]),
            schema.EntityKind(id="nn_module", child_kinds=["source_operator"]),
        ],
    )

    projected = project_entity_breakdowns(result)
    values = {
        breakdown.entity_id: breakdown.metrics[0].value
        for breakdown in projected.breakdowns
    }

    assert values == {
        "chain-a": 2,
        "chain-b": 3,
        "frame": 2,
        "module": 5,
        "op-a": 2,
        "op-b": 3,
    }


def test_overlapping_hierarchies_are_revisited_until_shared_state_converges() -> None:
    """A later hierarchy can populate a shared leaf for an earlier hierarchy."""
    metric = _metric(7, aggregation=schema.AggregationType.MAX)
    result = _result(
        [
            schema.Entity(
                id="target", kind="a_view", name="target", child_ids=["source"]
            ),
            schema.Entity(
                id="donor", kind="z_donor", name="donor", child_ids=["middle"]
            ),
            schema.Entity(
                id="middle", kind="z_middle", name="middle", child_ids=["source"]
            ),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [_breakdown("donor", metric)],
        [
            schema.EntityKind(id="a_view", child_kinds=["source_operator"]),
            schema.EntityKind(id="z_donor", child_kinds=["z_middle"]),
            schema.EntityKind(
                id="z_middle",
                parent_kinds=["z_donor"],
                child_kinds=["source_operator"],
            ),
        ],
    )

    projected = project_entity_breakdowns(result)

    for entity_id in ("middle", "source", "target"):
        assert [
            breakdown
            for breakdown in projected.breakdowns
            if breakdown.entity_id == entity_id
        ] == [_breakdown(entity_id, metric)]


def _cross_hierarchy_conflict_result(*, reverse: bool = False) -> schema.Result:
    """Return two hierarchy-local values competing for one shared source."""
    entities = [
        schema.Entity(id="a-root", kind="a_root", name="a", child_ids=["a-local"]),
        schema.Entity(id="a-local", kind="a_view", name="local", child_ids=["source"]),
        schema.Entity(id="target", kind="a_view", name="target", child_ids=["source"]),
        schema.Entity(id="b-root", kind="b_root", name="b", child_ids=["b-local"]),
        schema.Entity(
            id="b-local", kind="b_view", name="external", child_ids=["source"]
        ),
        schema.Entity(id="source", kind="source_operator", name="source"),
    ]
    breakdowns = [
        _breakdown("a-root", _metric(10, aggregation=schema.AggregationType.MAX)),
        _breakdown("b-root", _metric(20, aggregation=schema.AggregationType.MAX)),
    ]
    if reverse:
        entities.reverse()
        breakdowns.reverse()
    return _result(
        entities,
        breakdowns,
        [
            schema.EntityKind(id="a_root", child_kinds=["a_view"]),
            schema.EntityKind(
                id="a_view",
                parent_kinds=["a_root"],
                child_kinds=["source_operator"],
            ),
            schema.EntityKind(id="b_root", child_kinds=["b_view"]),
            schema.EntityKind(
                id="b_view",
                parent_kinds=["b_root"],
                child_kinds=["source_operator"],
            ),
        ],
    )


def test_conflict_in_a_shared_source_invalidates_dependent_hierarchy_values() -> None:
    """A target must not retain one hierarchy's value when its source conflicts."""
    projected = project_entity_breakdowns(_cross_hierarchy_conflict_result())

    assert not any(
        breakdown.entity_id in {"source", "target"}
        for breakdown in projected.breakdowns
    )


def test_exact_hierarchy_candidate_outranks_residual_candidate() -> None:
    """Cross-hierarchy reconciliation should retain the globally stronger value."""
    exact = _metric(10, aggregation=schema.AggregationType.MAX)
    result = _result(
        [
            schema.Entity(
                id="target",
                kind="source_operator",
                name="target",
                child_ids=["a", "b"],
            ),
            schema.Entity(
                id="exact-root",
                kind="a_root",
                name="exact root",
                child_ids=["exact"],
            ),
            schema.Entity(
                id="exact", kind="a_view", name="exact", child_ids=["a", "b"]
            ),
            schema.Entity(
                id="residual-root",
                kind="b_root",
                name="residual root",
                child_ids=["residual"],
            ),
            schema.Entity(
                id="residual",
                kind="b_view",
                name="residual",
                child_ids=["a", "b", "extra"],
            ),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
            schema.Entity(id="extra", kind="generated", name="extra"),
        ],
        [
            _breakdown("exact-root", exact),
            _breakdown(
                "residual-root",
                _metric(20, aggregation=schema.AggregationType.MAX),
            ),
        ],
        [
            schema.EntityKind(id="a_root", child_kinds=["a_view"]),
            schema.EntityKind(
                id="a_view",
                parent_kinds=["a_root"],
                child_kinds=["source_operator", "operator"],
            ),
            schema.EntityKind(id="b_root", child_kinds=["b_view"]),
            schema.EntityKind(
                id="b_view",
                parent_kinds=["b_root"],
                child_kinds=["source_operator", "operator", "generated"],
            ),
            schema.EntityKind(
                id="source_operator",
                parent_kinds=["a_view", "b_view"],
                child_kinds=["operator"],
            ),
        ],
    )

    assert _projected_for(result, "target") == [_breakdown("target", exact)]


def test_target_conflict_is_retried_after_stronger_inferred_copy_appears() -> None:
    """A transient conflict should not permanently suppress a later exact copy."""
    result = _result(
        [
            schema.Entity(
                id="target",
                kind="target_view",
                name="target",
                child_ids=["broad", "leaf-c"],
            ),
            schema.Entity(
                id="broad",
                kind="code_stack",
                name="broad",
                child_ids=["leaf-a", "leaf-b"],
            ),
            schema.Entity(id="leaf-a", kind="source_operator", name="a"),
            schema.Entity(
                id="bridge",
                kind="bridge_view",
                name="bridge",
                child_ids=["leaf-b", "leaf-c"],
            ),
            schema.Entity(
                id="c-donor",
                kind="code_stack",
                name="c donor",
                child_ids=["leaf-c"],
            ),
            schema.Entity(id="leaf-b", kind="source_operator", name="b"),
            schema.Entity(id="leaf-c", kind="source_operator", name="c"),
        ],
        [
            _breakdown("broad", _metric(9, aggregation=schema.AggregationType.SUM)),
            _breakdown("c-donor", _metric(16, aggregation=schema.AggregationType.SUM)),
            _breakdown("leaf-b", _metric(18, aggregation=schema.AggregationType.MAX)),
            _breakdown("leaf-a", _metric(16, aggregation=schema.AggregationType.SUM)),
        ],
    )

    assert _projected_for(result, "bridge") == [
        _breakdown("bridge", _metric(25, aggregation=schema.AggregationType.SUM))
    ]
    assert _projected_for(result, "target") == [
        _breakdown("target", _metric(25, aggregation=schema.AggregationType.SUM))
    ]


def test_projection_is_invariant_to_semantically_irrelevant_entity_ids() -> None:
    """Entity IDs must not determine which complete derivation is selected."""
    semantic_entities: list[tuple[str, str, list[str]]] = [
        ("target", "code_stack", ["broad", "x", "z"]),
        ("broad", "code_stack", ["x", "y"]),
        ("x", "source_operator", []),
        ("y", "source_operator", []),
        ("z", "source_operator", []),
    ]
    semantic_breakdowns: list[tuple[str, schema.Metric]] = [
        ("x", _metric(14, aggregation=schema.AggregationType.MAX)),
        ("y", _metric(16, aggregation=schema.AggregationType.SUM)),
        ("z", _metric(19, aggregation=schema.AggregationType.SUM)),
        ("broad", _metric(7, aggregation=schema.AggregationType.SUM)),
    ]

    def project_with_ids(ids: dict[str, str]) -> dict[str, int | float | None]:
        result = _result(
            [
                schema.Entity(
                    id=ids[name],
                    kind=kind,
                    name=name,
                    child_ids=[ids[child] for child in children],
                )
                for name, kind, children in semantic_entities
            ],
            [_breakdown(ids[name], metric) for name, metric in semantic_breakdowns],
        )
        names_by_id = {entity.id: entity.name for entity in result.entities}
        return {
            names_by_id[breakdown.entity_id]: breakdown.metrics[0].value
            for breakdown in project_entity_breakdowns(result).breakdowns
        }

    ordinary_ids = {
        "target": "e0",
        "broad": "e1",
        "x": "e2",
        "y": "e3",
        "z": "e4",
    }
    relabelled_ids = {
        "target": "re3",
        "broad": "re2",
        "x": "re4",
        "y": "re1",
        "z": "re0",
    }

    ordinary = project_with_ids(ordinary_ids)
    assert ordinary["target"] == 26
    assert project_with_ids(relabelled_ids) == ordinary


def test_equal_scope_targets_preserve_sealed_frontier_priority() -> None:
    """Raw alternatives must not override the sealed result for peer targets."""
    entities = [
        schema.Entity(
            id=target_id,
            kind=f"target-{index}",
            name=target_id,
            child_ids=["a", "b"],
        )
        for index, target_id in enumerate(("target-a", "target-b"))
    ]
    entities.extend(
        [
            schema.Entity(id="donor-a", kind="donor", name="a", child_ids=["a"]),
            schema.Entity(id="donor-b0", kind="donor", name="b0", child_ids=["b"]),
            schema.Entity(id="donor-b1", kind="donor", name="b1", child_ids=["b"]),
            schema.Entity(id="a", kind="source_operator", name="a"),
            schema.Entity(id="b", kind="source_operator", name="b"),
        ]
    )
    result = _result(
        entities,
        [
            _breakdown(
                "donor-a",
                _metric(5, aggregation=schema.AggregationType.SUM),
            ),
            _breakdown(
                "donor-b0",
                _metric(2, aggregation=schema.AggregationType.SUM),
            ),
            _breakdown(
                "donor-b1",
                _metric(2, aggregation=schema.AggregationType.SUM),
            ),
        ],
    )

    projected = project_entity_breakdowns(result)

    for target_id in ("target-a", "target-b"):
        assert [
            breakdown.metrics[0].value
            for breakdown in projected.breakdowns
            if breakdown.entity_id == target_id
        ] == [9]


def _equivalent_recipe_target_values(
    ids: dict[str, str],
) -> dict[str, float | int | None]:
    """Project equivalent recipe targets and return values by semantic name."""
    entities = [
        schema.Entity(
            id=ids["t0"],
            kind="peer",
            name="t0",
            child_ids=[ids["a"], ids["b"]],
        ),
        schema.Entity(
            id=ids["t1"],
            kind="peer",
            name="t1",
            child_ids=[ids["a"], ids["b"]],
        ),
        schema.Entity(
            id=ids["whole"],
            kind="whole",
            name="whole",
            child_ids=[ids["a"], ids["b"], ids["c"]],
        ),
        schema.Entity(
            id=ids["exact"],
            kind=schema.ENTITY_KIND_CODE_STACK,
            name="exact",
            child_ids=[ids["a"], ids["b"]],
        ),
        *[
            schema.Entity(
                id=ids[f"chain-{name}"],
                kind="chain",
                name=f"chain-{name}",
                child_ids=[ids[name]],
            )
            for name in ("a", "b", "c")
        ],
        *[
            schema.Entity(
                id=ids[name],
                kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                name=name,
            )
            for name in ("a", "b", "c")
        ],
    ]
    result = _result(
        entities,
        [
            *[
                _breakdown(
                    ids[f"chain-{name}"],
                    _metric(value, aggregation=schema.AggregationType.SUM),
                )
                for name, value in (("a", 2), ("b", 3), ("c", 11))
            ],
            _breakdown(
                ids["exact"],
                _metric(7, aggregation=schema.AggregationType.MAX),
            ),
        ],
    )
    names_by_id = {entity.id: entity.name for entity in result.entities}
    return {
        names_by_id[breakdown.entity_id]: breakdown.metrics[0].value
        for breakdown in project_entity_breakdowns(result).breakdowns
    }


def test_equivalent_recipe_targets_preserve_self_exclusion_symmetrically() -> None:
    """Equivalent t0/t1 producers must have the same recipe exposure semantics."""
    semantic_ids = (
        "t0",
        "t1",
        "whole",
        "exact",
        "chain-a",
        "chain-b",
        "chain-c",
        "a",
        "b",
        "c",
    )
    values = _equivalent_recipe_target_values({name: name for name in semantic_ids})

    assert values["whole"] == 16
    assert values["t0"] == 5
    assert values["t1"] == 5


def test_equivalent_recipe_targets_are_invariant_to_id_renaming() -> None:
    """Renaming t0/t1 must not choose which equivalent target receives output."""
    semantic_ids = (
        "t0",
        "t1",
        "whole",
        "exact",
        "chain-a",
        "chain-b",
        "chain-c",
        "a",
        "b",
        "c",
    )
    ordinary_ids = {name: name for name in semantic_ids}
    renamed_ids = {
        "t0": "z-target",
        "t1": "a-target",
        "whole": "whole-renamed",
        "exact": "exact-renamed",
        "chain-a": "chain-z",
        "chain-b": "chain-x",
        "chain-c": "chain-y",
        "a": "leaf-c",
        "b": "leaf-a",
        "c": "leaf-b",
    }

    assert _equivalent_recipe_target_values(renamed_ids) == (
        _equivalent_recipe_target_values(ordinary_ids)
    )


def test_conflicting_same_pass_candidates_are_order_independent() -> None:
    """A target must not commit one candidate before its competitor is known."""
    forward = project_entity_breakdowns(_cross_hierarchy_conflict_result())
    backward = project_entity_breakdowns(_cross_hierarchy_conflict_result(reverse=True))

    for projected in (forward, backward):
        assert not any(
            breakdown.entity_id in {"source", "target"}
            for breakdown in projected.breakdowns
        )


def test_projection_uses_inferred_figures_to_reach_a_fixed_point() -> None:
    """Larger represented leaf sets should build on smaller inferred figures."""
    result = _result(
        [
            schema.Entity(id="ab", kind="group", name="ab", child_ids=["a", "b"]),
            schema.Entity(
                id="abc", kind="group", name="abc", child_ids=["a", "b", "c"]
            ),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
            schema.Entity(id="c", kind="operator", name="c"),
        ],
        [
            _breakdown("a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("b", _metric(3, aggregation=schema.AggregationType.SUM)),
            _breakdown("c", _metric(4, aggregation=schema.AggregationType.SUM)),
        ],
    )

    projected = project_entity_breakdowns(result)
    values = {
        breakdown.entity_id: breakdown.metrics[0].value
        for breakdown in projected.breakdowns
    }

    assert values["ab"] == 5
    assert values["abc"] == 9


def test_existing_target_breakdown_remains_authoritative() -> None:
    """Projection should never replace or augment an entity with existing figures."""
    existing = _breakdown("target", _metric(99), breakdown_id="authoritative")
    result = _result(
        [
            schema.Entity(id="donor", kind="view", name="donor", child_ids=["leaf"]),
            schema.Entity(id="target", kind="view", name="target", child_ids=["leaf"]),
            schema.Entity(id="leaf", kind="operator", name="leaf"),
        ],
        [_breakdown("donor", _metric(5)), existing],
    )

    projected = project_entity_breakdowns(result)

    assert [item for item in projected.breakdowns if item.entity_id == "target"] == [
        existing
    ]


def test_metric_and_breakdown_compatibility_is_conservative() -> None:
    """Only numeric metrics with matching identities and metadata should aggregate."""
    warm = {"scenario": "warm"}
    result = _result(
        [
            schema.Entity(id="root", kind="group", name="root", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown(
                "a",
                _metric(
                    2,
                    qualifiers={"engine": "npu"},
                    aggregation="sum",
                    samples=2,
                ),
                _metric(10, name="wrong-unit", unit="bytes"),
                _metric(1, name="wrong-qualifier", qualifiers={"phase": "a"}),
                _metric(8, name="unavailable-on-peer"),
                _metric(9, name="wrong-aggregation", aggregation="sum"),
                qualifiers=warm,
            ),
            _breakdown(
                "b",
                _metric(
                    3,
                    qualifiers={"engine": "npu"},
                    aggregation="sum",
                    samples=4,
                ),
                _metric(20, name="wrong-unit", unit="cycles"),
                _metric(1, name="wrong-qualifier", qualifiers={"phase": "b"}),
                _metric(None, name="unavailable-on-peer"),
                _metric(10, name="wrong-aggregation", aggregation="mean"),
                qualifiers=warm,
            ),
            _breakdown("a", _metric(100), qualifiers={"scenario": "cold"}),
        ],
    )

    assert _projected_for(result, "root") == [
        _breakdown(
            "root",
            _metric(
                5,
                qualifiers={"engine": "npu"},
                aggregation="sum",
                samples=6,
            ),
            qualifiers=warm,
        )
    ]


def test_projected_order_is_deterministic_and_duplicates_are_collapsed() -> None:
    """All returned breakdowns should be unique and have stable ordering."""
    entities = [
        schema.Entity(id="z-target", kind="view", name="z", child_ids=["leaf"]),
        schema.Entity(id="a-target", kind="view", name="a", child_ids=["leaf"]),
        schema.Entity(id="donor", kind="view", name="donor", child_ids=["leaf"]),
        schema.Entity(id="leaf", kind="operator", name="leaf"),
    ]
    authoritative = _breakdown(
        "donor",
        _metric(2, name="z-metric"),
        _metric(1, name="a-metric"),
        qualifiers={"z": 1},
    )
    reordered_duplicate = _breakdown(
        "donor",
        _metric(1, name="a-metric"),
        _metric(2, name="z-metric"),
        qualifiers={"z": 1},
    )
    result = _result(entities, [authoritative, reordered_duplicate])

    projected = project_entity_breakdowns(result)

    assert projected.breakdowns[0] is authoritative
    assert [item.entity_id for item in projected.breakdowns] == [
        "donor",
        "a-target",
        "leaf",
        "z-target",
    ]
    metric_names = [
        [metric.name for metric in item.metrics] for item in projected.breakdowns
    ]
    assert metric_names == [
        ["z-metric", "a-metric"],
        ["a-metric", "z-metric"],
        ["a-metric", "z-metric"],
        ["a-metric", "z-metric"],
    ]
    serialized = [item.to_dict() for item in projected.breakdowns]
    assert len(serialized) == len({repr(item) for item in serialized})

    reversed_result = _result(reversed(entities), [authoritative, reordered_duplicate])
    reversed_breakdowns = project_entity_breakdowns(reversed_result).breakdowns
    assert [item.to_dict() for item in reversed_breakdowns] == serialized


def test_projection_fails_fast_for_unresolved_entity_reference() -> None:
    """Projection must invoke shared graph validation before inference."""
    result = _result(
        [
            schema.Entity(
                id="donor",
                kind="malformed",
                name="donor",
                child_ids=["missing"],
            ),
            schema.Entity(id="target", kind="view", name="target", child_ids=["donor"]),
        ],
        [_breakdown("donor", _metric(5))],
    )

    with pytest.raises(SchemaValidationError, match="does not resolve"):
        project_entity_breakdowns(result)


def test_projection_fails_fast_for_entity_cycle() -> None:
    """Projection must reject a cyclic graph even without top-level validation."""
    result = _result(
        [
            schema.Entity(id="a", kind="cycle", name="a", child_ids=["b"]),
            schema.Entity(id="b", kind="cycle", name="b", child_ids=["a"]),
            schema.Entity(id="target", kind="view", name="target", child_ids=["a"]),
        ],
        [_breakdown("a", _metric(5))],
    )

    with pytest.raises(SchemaValidationError, match="directed cycle"):
        project_entity_breakdowns(result)


def test_conflicting_authoritative_breakdowns_are_not_deduplicated() -> None:
    """Only exact semantic duplicates should be removed."""
    first = _breakdown("donor", _metric(5))
    second = _breakdown("donor", _metric(6))
    result = _result(
        [
            schema.Entity(id="donor", kind="view", name="donor", child_ids=["leaf"]),
            schema.Entity(id="target", kind="view", name="target", child_ids=["leaf"]),
            schema.Entity(id="leaf", kind="operator", name="leaf"),
        ],
        [first, second],
    )

    projected = project_entity_breakdowns(result)

    assert projected.breakdowns == [first, second]
    assert _projected_for(result, "target") == []


def test_deep_hierarchy_is_resolved_without_recursion() -> None:
    """A hierarchy deeper than Python's recursion limit should remain supported."""
    depth = 1_200
    entities = [
        schema.Entity(
            id=f"node-{index}",
            kind="view",
            name=f"node-{index}",
            child_ids=[f"node-{index + 1}"] if index + 1 < depth else [],
        )
        for index in range(depth)
    ]
    entities.append(
        schema.Entity(
            id="target",
            kind="view",
            name="target",
            child_ids=[f"node-{depth - 1}"],
        )
    )
    result = _result(entities, [_breakdown("node-0", _metric(5))])

    assert _projected_for(result, "target") == [_breakdown("target", _metric(5))]


def test_projection_does_not_mutate_input() -> None:
    """Projection should preserve every nested value in the caller-owned result."""
    donor = _breakdown(
        "donor",
        _metric(5, qualifiers={"phase": "execute"}),
        qualifiers={"run": "warm"},
    )
    result = _result(
        [
            schema.Entity(id="donor", kind="view", name="donor", child_ids=["leaf"]),
            schema.Entity(id="target", kind="view", name="target", child_ids=["leaf"]),
            schema.Entity(id="leaf", kind="operator", name="leaf"),
        ],
        [donor],
    )
    before = result.to_dict()

    projected = project_entity_breakdowns(result)

    assert result.to_dict() == before
    assert projected is not result
    assert result.breakdowns == [donor]
    assert _projected_for(result, "target") == [
        _breakdown(
            "target",
            _metric(5, qualifiers={"phase": "execute"}),
            qualifiers={"run": "warm"},
        )
    ]


def test_exact_leaf_set_copy_allows_unsupported_aggregation_policy() -> None:
    """Exact-view projection must not depend on aggregate policy support."""
    metric = _metric(5, aggregation=schema.AggregationType.MAX, samples=2)
    result = _result(
        [
            schema.Entity(id="donor", kind="view", name="donor", child_ids=["leaf"]),
            schema.Entity(id="target", kind="view", name="target", child_ids=["leaf"]),
            schema.Entity(id="leaf", kind="operator", name="leaf"),
        ],
        [_breakdown("donor", metric)],
    )

    assert _projected_for(result, "target") == [_breakdown("target", metric)]


def test_missing_aggregation_is_not_assumed_additive() -> None:
    """Different leaf sets require an explicit supported aggregation policy."""
    result = _result(
        [
            schema.Entity(id="root", kind="group", name="root", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [_breakdown("a", _metric(2)), _breakdown("b", _metric(3))],
    )

    assert _projected_for(result, "root") == []


def test_single_sum_metric_is_copied_across_no_attributable_statistics_gap() -> None:
    """A lone contributor is copied; the uncovered leaf performs no arithmetic."""
    result = _result(
        [
            schema.Entity(
                id="root", kind="group", name="root", child_ids=["real", "neutral"]
            ),
            schema.Entity(id="real", kind="operator", name="real"),
            schema.Entity(id="neutral", kind="operator", name="neutral"),
        ],
        [
            _breakdown(
                "real",
                _metric(7, aggregation=schema.AggregationType.SUM, samples=4),
            )
        ],
    )

    projected = project_entity_breakdowns(result)

    assert [item for item in projected.breakdowns if item.entity_id == "root"] == [
        _breakdown(
            "root",
            _metric(7, aggregation=schema.AggregationType.SUM, samples=4),
        )
    ]


def test_unusable_breakdowns_account_for_leaves_without_creating_an_anchor() -> None:
    """Covered leaves remain accounted even when their figures are unusable."""
    result = _result(
        [
            schema.Entity(
                id="root",
                kind="group",
                name="root",
                child_ids=["neutral-a", "neutral-b"],
            ),
            schema.Entity(id="neutral-a", kind="operator", name="a"),
            schema.Entity(id="neutral-b", kind="operator", name="b"),
        ],
        [
            schema.Breakdown(entity_id="neutral-a", metrics=[]),
            schema.Breakdown(entity_id="neutral-b", metrics=[]),
        ],
    )

    projected = project_entity_breakdowns(result)

    assert not any(item.entity_id == "root" for item in projected.breakdowns)


def test_aggregate_only_breakdown_accounts_for_descendant_without_direct_data() -> None:
    """Aggregate authoritative coverage prevents a descendant becoming a free gap."""
    result = _result(
        [
            schema.Entity(
                id="root", kind="root", name="root", child_ids=["a", "group"]
            ),
            schema.Entity(id="group", kind="group", name="group", child_ids=["b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown("a", _metric(3)),
            _breakdown("group", _metric(None)),
        ],
    )

    assert _projected_for(result, "root") == []


def test_conflicting_authoritative_breakdowns_still_account_for_their_leaves() -> None:
    """Conflicting figures account for leaves without enabling projection."""
    result = _result(
        [
            schema.Entity(id="root", kind="root", name="root", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown("a", _metric(3)),
            _breakdown("b", _metric(4)),
            _breakdown("b", _metric(5)),
        ],
    )

    assert _projected_for(result, "root") == []


def test_contributor_without_target_intersection_has_no_attribution_anchor() -> None:
    """An extras-only donor must not create a projection without a target leaf."""
    result = _result(
        [
            schema.Entity(id="donor", kind="donor", name="donor", child_ids=["extra"]),
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["target-leaf"]
            ),
            schema.Entity(id="extra", kind="operator", name="extra"),
            schema.Entity(id="target-leaf", kind="operator", name="target-leaf"),
        ],
        [_breakdown("donor", _metric(7))],
    )

    assert _projected_for(result, "target") == []


def test_single_contributor_with_uncovered_gap_copies_unsupported_policy() -> None:
    """One complete attribution is a copy, not arithmetic over uncovered leaves."""
    metric = _metric(7, aggregation=schema.AggregationType.MAX)
    result = _result(
        [
            schema.Entity(
                id="root", kind="group", name="root", child_ids=["real", "neutral"]
            ),
            schema.Entity(id="real", kind="operator", name="real"),
            schema.Entity(id="neutral", kind="operator", name="neutral"),
        ],
        [_breakdown("real", metric)],
    )

    projected = project_entity_breakdowns(result)

    assert [item for item in projected.breakdowns if item.entity_id == "root"] == [
        _breakdown("root", metric)
    ]


def test_single_contributor_with_uncontested_extra_is_attributed() -> None:
    """A stack may retain unclaimed chain work outside its source coverage."""
    result = _result(
        [
            schema.Entity(
                id="chain",
                kind="chain",
                name="chain",
                child_ids=["op-a", "op-b", "extra"],
            ),
            schema.Entity(
                id="stack",
                kind=schema.ENTITY_KIND_CODE_STACK,
                name="stack",
                child_ids=["op-a", "op-b"],
            ),
            schema.Entity(id="op-a", kind=schema.ENTITY_KIND_SOURCE_OPERATOR, name="a"),
            schema.Entity(id="op-b", kind=schema.ENTITY_KIND_SOURCE_OPERATOR, name="b"),
            schema.Entity(
                id="extra",
                kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                name="extra",
            ),
        ],
        [_breakdown("chain", _metric(11, aggregation=schema.AggregationType.MAX))],
        [
            schema.EntityKind(
                id="chain",
                child_kinds=[schema.ENTITY_KIND_SOURCE_OPERATOR],
            )
        ],
    )

    assert _projected_for(result, "stack") == [
        _breakdown("stack", _metric(11, aggregation=schema.AggregationType.MAX))
    ]


def test_accounted_extra_is_blocked_when_claimed() -> None:
    """A donor extra is attributable because its authoritative footprint covers it."""
    result = _result(
        [
            schema.Entity(
                id="donor",
                kind="generated_view",
                name="donor",
                child_ids=["op", "neutral"],
            ),
            schema.Entity(id="target", kind="view", name="target", child_ids=["op"]),
            schema.Entity(
                id="claimant", kind="view", name="claimant", child_ids=["neutral"]
            ),
            schema.Entity(id="op", kind="operator", name="op"),
            schema.Entity(id="neutral", kind="operator", name="neutral"),
        ],
        [_breakdown("donor", _metric(8))],
    )

    projected = project_entity_breakdowns(result)

    assert not any(item.entity_id == "target" for item in projected.breakdowns)


def test_same_kind_claimant_blocks_unrelated_extra() -> None:
    """Same-kind competition is global and does not require a shared ancestor."""
    result = _result(
        [
            schema.Entity(
                id="donor", kind="donor", name="donor", child_ids=["op", "extra"]
            ),
            schema.Entity(id="target", kind="view", name="target", child_ids=["op"]),
            schema.Entity(
                id="claimant", kind="view", name="claimant", child_ids=["extra"]
            ),
            schema.Entity(id="op", kind="operator", name="op"),
            schema.Entity(id="extra", kind="operator", name="extra"),
        ],
        [_breakdown("donor", _metric(9))],
    )

    assert _projected_for(result, "target") == []


@pytest.mark.parametrize("claimant_kind", ["sibling", "ancestor"])
def test_same_hierarchy_claimant_blocks_extra(claimant_kind: str) -> None:
    """A sibling through a common parent or an ancestor may contest an extra."""
    if claimant_kind == "sibling":
        hierarchy = [
            schema.Entity(
                id="root", kind="root", name="root", child_ids=["target", "claimant"]
            ),
            schema.Entity(id="target", kind="target", name="target", child_ids=["op"]),
            schema.Entity(
                id="claimant", kind="claimant", name="claimant", child_ids=["extra"]
            ),
        ]
    else:
        hierarchy = [
            schema.Entity(
                id="claimant",
                kind="claimant",
                name="claimant",
                child_ids=["target", "extra"],
            ),
            schema.Entity(id="target", kind="target", name="target", child_ids=["op"]),
        ]

    result = _result(
        [
            schema.Entity(
                id="donor", kind="donor", name="donor", child_ids=["op", "extra"]
            ),
            *hierarchy,
            schema.Entity(id="op", kind="operator", name="op"),
            schema.Entity(id="extra", kind="operator", name="extra"),
        ],
        [_breakdown("donor", _metric(9))],
    )

    assert _projected_for(result, "target") == []


def test_different_kind_unrelated_claimant_does_not_block_extra() -> None:
    """Different-kind entities in independent hierarchies do not compete."""
    result = _result(
        [
            schema.Entity(
                id="donor", kind="code_stack", name="donor", child_ids=["op", "extra"]
            ),
            schema.Entity(id="target", kind="target", name="target", child_ids=["op"]),
            schema.Entity(
                id="claimant", kind="other", name="claimant", child_ids=["extra"]
            ),
            schema.Entity(id="op", kind="source_operator", name="op"),
            schema.Entity(id="extra", kind="source_operator", name="extra"),
        ],
        [_breakdown("donor", _metric(9))],
    )

    assert _projected_for(result, "target") == [_breakdown("target", _metric(9))]


def test_independent_targets_may_both_receive_one_contributor() -> None:
    """Independent hierarchy views may intentionally repeat the same figures."""
    result = _result(
        [
            schema.Entity(
                id="donor", kind="code_stack", name="donor", child_ids=["op", "extra"]
            ),
            schema.Entity(
                id="first", kind="first_view", name="first", child_ids=["op"]
            ),
            schema.Entity(
                id="second", kind="second_view", name="second", child_ids=["op"]
            ),
            schema.Entity(id="op", kind="source_operator", name="op"),
            schema.Entity(id="extra", kind="source_operator", name="extra"),
        ],
        [_breakdown("donor", _metric(12))],
    )

    projected = project_entity_breakdowns(result)
    values = {item.entity_id: item.metrics[0].value for item in projected.breakdowns}

    assert values["first"] == 12
    assert values["second"] == 12


def test_common_parent_receives_contributor_that_children_cannot() -> None:
    """Exact parent attribution is valid while sibling claims block each child."""
    result = _result(
        [
            schema.Entity(
                id="donor", kind="donor", name="donor", child_ids=["left", "right"]
            ),
            schema.Entity(
                id="parent",
                kind="parent",
                name="parent",
                child_ids=["left_view", "right_view"],
            ),
            schema.Entity(id="left_view", kind="left", name="left", child_ids=["left"]),
            schema.Entity(
                id="right_view", kind="right", name="right", child_ids=["right"]
            ),
            schema.Entity(id="left", kind="operator", name="left"),
            schema.Entity(id="right", kind="operator", name="right"),
        ],
        [_breakdown("donor", _metric(20, aggregation=schema.AggregationType.MAX))],
    )

    projected = project_entity_breakdowns(result)

    assert [item for item in projected.breakdowns if item.entity_id == "parent"] == [
        _breakdown("parent", _metric(20, aggregation=schema.AggregationType.MAX))
    ]
    assert not any(
        item.entity_id in {"left_view", "right_view"} for item in projected.breakdowns
    )


def test_multiple_sum_contributors_may_each_have_uncontested_extras() -> None:
    """Residual SUM attribution unions disjoint donor footprints."""
    result = _result(
        [
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(id="first", kind="first", name="first", child_ids=["a", "x"]),
            schema.Entity(
                id="second", kind="second", name="second", child_ids=["b", "y"]
            ),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
            schema.Entity(id="x", kind="generated", name="x"),
            schema.Entity(id="y", kind="generated", name="y"),
        ],
        [
            _breakdown(
                "first", _metric(2, aggregation=schema.AggregationType.SUM, samples=1)
            ),
            _breakdown(
                "second", _metric(3, aggregation=schema.AggregationType.SUM, samples=2)
            ),
        ],
    )

    assert _projected_for(result, "target") == [
        _breakdown(
            "target",
            _metric(5, aggregation=schema.AggregationType.SUM, samples=3),
        )
    ]


def test_distinct_origins_remain_additive_after_residual_copy() -> None:
    """A copied origin remains distinct from another additive origin."""
    result = _result(
        [
            schema.Entity(
                id="first", kind="code_stack", name="first", child_ids=["a", "shared"]
            ),
            schema.Entity(
                id="inferred", kind="inferred", name="inferred", child_ids=["a"]
            ),
            schema.Entity(
                id="second", kind="code_stack", name="second", child_ids=["b", "shared"]
            ),
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(id="a", kind="source_operator", name="a"),
            schema.Entity(id="b", kind="source_operator", name="b"),
            schema.Entity(id="shared", kind="source_operator", name="shared"),
        ],
        [
            _breakdown("first", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("second", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
    )

    projected = project_entity_breakdowns(result)

    # Two chains may each include the same uncontested generated work while
    # measuring distinct scheduled contributions. Copying the first chain to an
    # intermediate view must retain its origin ID, not turn the shared graph
    # entity into a false accounting collision with the second chain.
    assert [item for item in projected.breakdowns if item.entity_id == "inferred"] == [
        _breakdown("inferred", _metric(2, aggregation=schema.AggregationType.SUM))
    ]
    assert [item for item in projected.breakdowns if item.entity_id == "target"] == [
        _breakdown("target", _metric(5, aggregation=schema.AggregationType.SUM))
    ]


def test_equal_leaf_targets_can_have_different_attribution_compatibility() -> None:
    """Residual compatibility is scoped to each target hierarchy, not leaf cache."""
    result = _result(
        [
            schema.Entity(
                id="donor", kind="code_stack", name="donor", child_ids=["op", "extra"]
            ),
            schema.Entity(
                id="root", kind="root", name="root", child_ids=["related", "claimant"]
            ),
            schema.Entity(
                id="related", kind="related", name="related", child_ids=["op"]
            ),
            schema.Entity(
                id="claimant", kind="claimant", name="claimant", child_ids=["extra"]
            ),
            schema.Entity(
                id="independent",
                kind="independent",
                name="independent",
                child_ids=["op"],
            ),
            schema.Entity(id="op", kind="source_operator", name="op"),
            schema.Entity(id="extra", kind="source_operator", name="extra"),
        ],
        [_breakdown("donor", _metric(13))],
    )

    projected = project_entity_breakdowns(result)

    assert not any(item.entity_id == "related" for item in projected.breakdowns)
    assert [
        item for item in projected.breakdowns if item.entity_id == "independent"
    ] == [_breakdown("independent", _metric(13))]


def test_residual_copy_with_fewer_extras_outranks_broader_residual_copy() -> None:
    """A residual copy with fewer terminal extras is the stronger attribution."""
    result = _result(
        [
            schema.Entity(
                id="target", kind="code_stack", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(
                id="segment",
                kind="segment",
                name="segment",
                child_ids=["chain-112", "chain-108", "chain-106"],
            ),
            schema.Entity(
                id="chain-112",
                kind="chain",
                name="chain-112",
                parent_ids=["segment"],
                child_ids=["a", "b", "c"],
            ),
            schema.Entity(
                id="chain-108",
                kind="chain",
                name="chain-108",
                parent_ids=["segment"],
            ),
            schema.Entity(
                id="chain-106",
                kind="chain",
                name="chain-106",
                parent_ids=["segment"],
            ),
            schema.Entity(id="a", kind="source_operator", name="a"),
            schema.Entity(id="b", kind="source_operator", name="b"),
            schema.Entity(id="c", kind="source_operator", name="c"),
        ],
        [
            _breakdown(
                "chain-112", _metric(524, aggregation=schema.AggregationType.SUM)
            ),
            _breakdown(
                "chain-108", _metric(207, aggregation=schema.AggregationType.SUM)
            ),
            _breakdown(
                "chain-106", _metric(1267, aggregation=schema.AggregationType.SUM)
            ),
        ],
        [
            schema.EntityKind(id="segment", child_kinds=["chain"]),
            schema.EntityKind(
                id="chain",
                parent_kinds=["segment"],
                child_kinds=["source_operator"],
            ),
        ],
    )

    assert _projected_for(result, "target") == [
        _breakdown("target", _metric(524, aggregation=schema.AggregationType.SUM))
    ]


def test_equal_extra_counts_preserve_same_priority_conflict() -> None:
    """Extra count should not choose between equally broad residual copies."""
    result = _result(
        [
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(
                id="first", kind="code_stack", name="first", child_ids=["a", "b", "x"]
            ),
            schema.Entity(
                id="second",
                kind="code_stack",
                name="second",
                child_ids=["a", "b", "y"],
            ),
            schema.Entity(id="a", kind="source_operator", name="a"),
            schema.Entity(id="b", kind="source_operator", name="b"),
            schema.Entity(id="x", kind="source_operator", name="x"),
            schema.Entity(id="y", kind="source_operator", name="y"),
        ],
        [
            _breakdown("first", _metric(5, aggregation=schema.AggregationType.MAX)),
            _breakdown("second", _metric(7, aggregation=schema.AggregationType.MAX)),
        ],
    )

    assert _projected_for(result, "target") == []


def test_no_extra_derivation_wins_over_conflicting_residual_copy() -> None:
    """A complete no-extra SUM outranks a residual result with another value."""
    result = _result(
        [
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(
                id="residual",
                kind="residual",
                name="residual",
                child_ids=["a", "b", "x"],
            ),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
            schema.Entity(id="x", kind="generated", name="x"),
        ],
        [
            _breakdown("a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("b", _metric(3, aggregation=schema.AggregationType.SUM)),
            _breakdown("residual", _metric(99, aggregation=schema.AggregationType.MAX)),
        ],
    )

    assert _projected_for(result, "target") == [
        _breakdown("target", _metric(5, aggregation=schema.AggregationType.SUM))
    ]


def test_nine_exact_groups_outrank_lower_priority_residual() -> None:
    """An exhaustive no-extra search must resolve before residual fallback."""
    group_count = 9
    leaf_pairs = [(f"a-{index}", f"b-{index}") for index in range(group_count)]
    leaf_ids = [leaf_id for pair in leaf_pairs for leaf_id in pair]
    result = _result(
        [
            schema.Entity(
                id="target",
                kind="target",
                name="target",
                child_ids=leaf_ids,
            ),
            schema.Entity(
                id="residual",
                kind=schema.ENTITY_KIND_CODE_STACK,
                name="residual",
                child_ids=[*leaf_ids, "extra"],
            ),
            *[
                schema.Entity(
                    id=f"exact-{index}",
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name=f"exact-{index}",
                    child_ids=list(pair),
                )
                for index, pair in enumerate(leaf_pairs)
            ],
            *[
                schema.Entity(
                    id=f"split-{leaf_id}",
                    kind=schema.ENTITY_KIND_CODE_STACK,
                    name=f"split-{leaf_id}",
                    child_ids=[leaf_id],
                )
                for leaf_id in leaf_ids
            ],
            *[
                schema.Entity(
                    id=leaf_id,
                    kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                    name=leaf_id,
                )
                for leaf_id in [*leaf_ids, "extra"]
            ],
        ],
        [
            *[
                _breakdown(
                    f"exact-{index}",
                    _metric(3, aggregation=schema.AggregationType.SUM),
                )
                for index in range(group_count)
            ],
            *[
                _breakdown(
                    f"split-{leaf_id}",
                    _metric(value, aggregation=schema.AggregationType.SUM),
                )
                for pair in leaf_pairs
                for leaf_id, value in zip(pair, (1, 2))
            ],
            _breakdown(
                "residual",
                _metric(999, aggregation=schema.AggregationType.MAX),
            ),
        ],
    )

    assert _projected_for(result, "target") == [
        _breakdown(
            "target",
            _metric(27, aggregation=schema.AggregationType.SUM),
        )
    ]


def test_authoritative_exact_match_outranks_no_extra_and_residual_derivations() -> None:
    """An exact MAX copy wins over conflicting SUM and residual values."""
    exact_metric = _metric(50, aggregation=schema.AggregationType.MAX)
    result = _result(
        [
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(
                id="exact", kind="code_stack", name="exact", child_ids=["a", "b"]
            ),
            schema.Entity(
                id="residual",
                kind="code_stack",
                name="residual",
                child_ids=["a", "b", "x"],
            ),
            schema.Entity(id="a", kind="source_operator", name="a"),
            schema.Entity(id="b", kind="source_operator", name="b"),
            schema.Entity(id="x", kind="source_operator", name="x"),
        ],
        [
            _breakdown("exact", exact_metric),
            _breakdown("a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("b", _metric(3, aggregation=schema.AggregationType.SUM)),
            _breakdown("residual", _metric(99, aggregation=schema.AggregationType.MAX)),
        ],
    )

    assert _projected_for(result, "target") == [_breakdown("target", exact_metric)]


def test_same_priority_sum_results_with_values_five_and_seven_are_ambiguous() -> None:
    """The competing SUM calculations 2 + 3 and 4 + 3 suppress projection."""
    result = _result(
        [
            schema.Entity(
                id="target", kind="target", name="target", child_ids=["a", "b"]
            ),
            schema.Entity(id="a-first", kind="a_first", name="a", child_ids=["a"]),
            schema.Entity(id="a-second", kind="a_second", name="a", child_ids=["a"]),
            schema.Entity(id="b-first", kind="b_first", name="b", child_ids=["b"]),
            schema.Entity(id="b-second", kind="b_second", name="b", child_ids=["b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown("a-first", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("a-second", _metric(4, aggregation=schema.AggregationType.SUM)),
            _breakdown("b-first", _metric(3, aggregation=schema.AggregationType.SUM)),
            _breakdown("b-second", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
    )

    assert _projected_for(result, "target") == []


def test_residual_projection_is_independent_of_entity_and_breakdown_order() -> None:
    """Normalized graphs and derivation enumeration have deterministic results."""
    entities = [
        schema.Entity(id="target", kind="target", name="target", child_ids=["a", "b"]),
        schema.Entity(id="first", kind="first", name="first", child_ids=["a", "x"]),
        schema.Entity(id="second", kind="second", name="second", child_ids=["b", "y"]),
        schema.Entity(id="a", kind="operator", name="a"),
        schema.Entity(id="b", kind="operator", name="b"),
        schema.Entity(id="x", kind="generated", name="x"),
        schema.Entity(id="y", kind="generated", name="y"),
    ]
    breakdowns = [
        _breakdown("first", _metric(2, aggregation=schema.AggregationType.SUM)),
        _breakdown("second", _metric(3, aggregation=schema.AggregationType.SUM)),
    ]

    forward = project_entity_breakdowns(_result(entities, breakdowns))
    backward = project_entity_breakdowns(
        _result(reversed(entities), reversed(breakdowns))
    )

    forward_projected = sorted(
        (item.to_dict() for item in forward.breakdowns if item.entity_id == "target"),
        key=repr,
    )
    backward_projected = sorted(
        (item.to_dict() for item in backward.breakdowns if item.entity_id == "target"),
        key=repr,
    )
    assert backward_projected == forward_projected


def test_source_line_view_deduplicates_two_stacks_with_the_same_origin() -> None:
    """Repeated provenance paths must not duplicate one accounting origin."""
    result = _result(
        [
            schema.Entity(id="chain", kind="chain", name="chain", child_ids=["source"]),
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:10",
                child_ids=["stack-a", "stack-b"],
            ),
            schema.Entity(
                id="stack-a",
                kind="code_stack",
                name="a",
                child_ids=["source"],
            ),
            schema.Entity(
                id="stack-b",
                kind="code_stack",
                name="b",
                child_ids=["source"],
            ),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [_breakdown("chain", _metric(5, aggregation=schema.AggregationType.SUM))],
        [
            *_CHAIN_SOURCE_ENTITY_KINDS,
            schema.EntityKind(id="source_line_view", child_kinds=["code_stack"]),
        ],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(5, aggregation=schema.AggregationType.SUM))
    ]


def test_source_line_view_uses_complete_nested_stack_instead_of_adding_child() -> None:
    """A complete root stack should subsume its narrower same-line child."""
    result = _result(
        [
            schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["source-a"]),
            schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["source-b"]),
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:20",
                child_ids=["stack-root", "stack-child"],
            ),
            schema.Entity(
                id="stack-root",
                kind="code_stack",
                name="root",
                child_ids=["stack-child", "source-b"],
            ),
            schema.Entity(
                id="stack-child",
                kind="code_stack",
                name="child",
                child_ids=["source-a"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [
            _breakdown("chain-a", _metric(4, aggregation=schema.AggregationType.SUM)),
            _breakdown("chain-b", _metric(6, aggregation=schema.AggregationType.SUM)),
        ],
        [
            *_CHAIN_SOURCE_ENTITY_KINDS,
            schema.EntityKind(id="source_line_view", child_kinds=["code_stack"]),
        ],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(10, aggregation=schema.AggregationType.SUM))
    ]


def test_source_line_view_sums_disjoint_stack_origins() -> None:
    """Separate stack contributions on one line should form an additive total."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:30",
                child_ids=["stack-a", "stack-b"],
            ),
            schema.Entity(
                id="stack-a",
                kind="code_stack",
                name="a",
                child_ids=["source-a"],
            ),
            schema.Entity(
                id="stack-b",
                kind="code_stack",
                name="b",
                child_ids=["source-b"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [
            _breakdown("stack-a", _metric(2, aggregation=schema.AggregationType.SUM)),
            _breakdown("stack-b", _metric(3, aggregation=schema.AggregationType.SUM)),
        ],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(5, aggregation=schema.AggregationType.SUM))
    ]


def test_source_line_view_sums_distinct_origins_with_overlapping_coverage() -> None:
    """Structural overlap must not merge independently owned stack measurements."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:40",
                child_ids=["stack-left", "stack-right"],
            ),
            schema.Entity(
                id="stack-left",
                kind="code_stack",
                name="left",
                child_ids=["source-a", "source-b"],
            ),
            schema.Entity(
                id="stack-right",
                kind="code_stack",
                name="right",
                child_ids=["source-b", "source-c"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
            schema.Entity(id="source-c", kind="source_operator", name="c"),
        ],
        [
            _breakdown(
                "stack-left", _metric(5, aggregation=schema.AggregationType.SUM)
            ),
            _breakdown(
                "stack-right", _metric(7, aggregation=schema.AggregationType.SUM)
            ),
        ],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(12, aggregation=schema.AggregationType.SUM))
    ]


def test_source_line_view_copies_complete_max_stack_over_narrow_sum_child() -> None:
    """An exact complete stack is a copy and outranks a narrower subtotal."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:50",
                child_ids=["stack-root", "stack-child"],
            ),
            schema.Entity(
                id="stack-root",
                kind="code_stack",
                name="root",
                child_ids=["stack-child", "source-b"],
            ),
            schema.Entity(
                id="stack-child",
                kind="code_stack",
                name="child",
                child_ids=["source-a"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [
            _breakdown(
                "stack-root", _metric(10, aggregation=schema.AggregationType.MAX)
            ),
            _breakdown(
                "stack-child", _metric(4, aggregation=schema.AggregationType.SUM)
            ),
        ],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(10, aggregation=schema.AggregationType.MAX))
    ]


def test_source_line_view_copies_one_measured_stack_across_unmeasured_sibling() -> None:
    """A stack without authoritative statistics does not block a safe line value."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:60",
                child_ids=["stack-measured", "stack-unmeasured"],
            ),
            schema.Entity(
                id="stack-measured",
                kind="code_stack",
                name="measured",
                child_ids=["source-a"],
            ),
            schema.Entity(
                id="stack-unmeasured",
                kind="code_stack",
                name="unmeasured",
                child_ids=["source-b"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [
            _breakdown(
                "stack-measured",
                _metric(2, aggregation=schema.AggregationType.SUM),
            )
        ],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(2, aggregation=schema.AggregationType.SUM))
    ]


def test_source_line_view_does_not_assume_implicit_stack_values_are_additive() -> None:
    """Multiple stack origins still require an explicit additive policy."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:70",
                child_ids=["stack-a", "stack-b"],
            ),
            schema.Entity(
                id="stack-a",
                kind="code_stack",
                name="a",
                child_ids=["source-a"],
            ),
            schema.Entity(
                id="stack-b",
                kind="code_stack",
                name="b",
                child_ids=["source-b"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [_breakdown("stack-a", _metric(2)), _breakdown("stack-b", _metric(3))],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == []


def test_adding_source_line_view_does_not_change_existing_projections() -> None:
    """A cross-cutting stack parent must not perturb established hierarchy values."""
    entities = [
        schema.Entity(id="chain-a", kind="chain", name="a", child_ids=["source-a"]),
        schema.Entity(id="chain-b", kind="chain", name="b", child_ids=["source-b"]),
        schema.Entity(
            id="stack-root",
            kind="code_stack",
            name="root",
            child_ids=["stack-child", "source-b"],
        ),
        schema.Entity(
            id="stack-child",
            kind="code_stack",
            name="child",
            child_ids=["source-a"],
        ),
        schema.Entity(
            id="module",
            kind="nn_module",
            name="module",
            child_ids=["source-a", "source-b"],
        ),
        schema.Entity(id="source-a", kind="source_operator", name="a"),
        schema.Entity(id="source-b", kind="source_operator", name="b"),
    ]
    breakdowns = [
        _breakdown("chain-a", _metric(4, aggregation=schema.AggregationType.SUM)),
        _breakdown("chain-b", _metric(6, aggregation=schema.AggregationType.SUM)),
    ]
    kinds = [
        *_CHAIN_SOURCE_ENTITY_KINDS,
        schema.EntityKind(id="nn_module", child_kinds=["source_operator"]),
    ]
    without_line = project_entity_breakdowns(_result(entities, breakdowns, kinds))
    with_line = project_entity_breakdowns(
        _result(
            [
                *entities,
                schema.Entity(
                    id="line",
                    kind="source_line_view",
                    name="model.py:80",
                    child_ids=["stack-root", "stack-child"],
                ),
            ],
            breakdowns,
            [
                *kinds,
                schema.EntityKind(id="source_line_view", child_kinds=["code_stack"]),
            ],
        )
    )

    assert [item.to_dict() for item in without_line.breakdowns] == [
        item.to_dict() for item in with_line.breakdowns if item.entity_id != "line"
    ]
    line_breakdowns = [
        item.to_dict() for item in with_line.breakdowns if item.entity_id == "line"
    ]
    assert line_breakdowns == [
        _breakdown(
            "line", _metric(10, aggregation=schema.AggregationType.SUM)
        ).to_dict()
    ]


def test_source_line_view_uses_known_equivalent_when_other_stack_is_missing() -> None:
    """An unmeasured equivalent provenance path must not hide a known figure."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:90",
                child_ids=["stack-known", "stack-missing"],
            ),
            schema.Entity(
                id="stack-known",
                kind="code_stack",
                name="known",
                child_ids=["source"],
            ),
            schema.Entity(
                id="stack-missing",
                kind="code_stack",
                name="missing",
                child_ids=["source"],
            ),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [_breakdown("stack-known", _metric(9, aggregation=schema.AggregationType.SUM))],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(9, aggregation=schema.AggregationType.SUM))
    ]


def test_source_line_view_rejects_incompatible_exact_stack_figures() -> None:
    """Equal values with incompatible policies remain conflicting alternatives."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:100",
                child_ids=["stack-sum", "stack-max"],
            ),
            schema.Entity(
                id="stack-sum",
                kind="code_stack",
                name="sum",
                child_ids=["source"],
            ),
            schema.Entity(
                id="stack-max",
                kind="code_stack",
                name="max",
                child_ids=["source"],
            ),
            schema.Entity(id="source", kind="source_operator", name="source"),
        ],
        [
            _breakdown("stack-sum", _metric(9, aggregation=schema.AggregationType.SUM)),
            _breakdown("stack-max", _metric(9, aggregation=schema.AggregationType.MAX)),
        ],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == []


def test_source_line_view_uses_measured_child_when_recursive_root_is_missing() -> None:
    """A missing broad stack should not block its safely attributable child."""
    result = _result(
        [
            schema.Entity(
                id="line",
                kind="source_line_view",
                name="model.py:110",
                child_ids=["stack-root", "stack-child"],
            ),
            schema.Entity(
                id="stack-root",
                kind="code_stack",
                name="root",
                child_ids=["stack-child", "source-b"],
            ),
            schema.Entity(
                id="stack-child",
                kind="code_stack",
                name="child",
                child_ids=["source-a"],
            ),
            schema.Entity(id="source-a", kind="source_operator", name="a"),
            schema.Entity(id="source-b", kind="source_operator", name="b"),
        ],
        [_breakdown("stack-child", _metric(4, aggregation=schema.AggregationType.SUM))],
        [schema.EntityKind(id="source_line_view", child_kinds=["code_stack"])],
    )

    assert _projected_for(result, "line") == [
        _breakdown("line", _metric(4, aggregation=schema.AggregationType.SUM))
    ]


def test_non_finite_aggregate_is_suppressed() -> None:
    """Finite input metrics must not produce an infinite projected metric."""
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    result = _result(
        [
            schema.Entity(id="line", kind="view", name="line", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown("a", _metric(maximum, aggregation=schema.AggregationType.SUM)),
            _breakdown("b", _metric(maximum, aggregation=schema.AggregationType.SUM)),
        ],
    )

    assert _projected_for(result, "line") == []


def test_incompatible_sample_metadata_is_not_aggregated() -> None:
    """Sample counts must be consistently present on every contributor."""
    result = _result(
        [
            schema.Entity(id="line", kind="view", name="line", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown(
                "a",
                _metric(2.25, aggregation=schema.AggregationType.SUM, samples=1),
            ),
            _breakdown(
                "b",
                _metric(3.5, aggregation=schema.AggregationType.SUM),
            ),
        ],
    )

    assert _projected_for(result, "line") == []


def test_aggregate_samples_must_fit_the_interoperable_integer_range() -> None:
    """Projected sample counts must remain exact in JSON and browser consumers."""
    maximum_safe_integer = 2**53 - 1
    result = _result(
        [
            schema.Entity(id="line", kind="view", name="line", child_ids=["a", "b"]),
            schema.Entity(id="a", kind="operator", name="a"),
            schema.Entity(id="b", kind="operator", name="b"),
        ],
        [
            _breakdown(
                "a",
                _metric(
                    2,
                    aggregation=schema.AggregationType.SUM,
                    samples=maximum_safe_integer,
                ),
            ),
            _breakdown(
                "b",
                _metric(3, aggregation=schema.AggregationType.SUM, samples=1),
            ),
        ],
    )

    assert _projected_for(result, "line") == []
