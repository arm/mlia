# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for generic core entity collapse and output postprocessing."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

import mlia.core.output_schema as schema
from mlia.core.entity_collapse import collapse_entities
from mlia.core.output_postprocessing import postprocess_standardized_output
from mlia.core.output_projection import project_entity_breakdowns
from mlia.core.output_validation import SchemaValidationError
from mlia.core.settings import ApplicationSettings, CollapseRule, FilteringSettings


def _entity(
    entity_id: str,
    *,
    kind: str = "node",
    parents: list[str] | None = None,
    children: list[str] | None = None,
    attributes: dict[str, object] | None = None,
) -> schema.Entity:
    return schema.Entity(
        id=entity_id,
        kind=kind,
        name=entity_id,
        parent_ids=parents or [],
        child_ids=children or [],
        attributes=attributes or {},
    )


def _result(
    entities: list[schema.Entity],
    breakdowns: list[schema.Breakdown] | None = None,
    checks: list[schema.Check] | None = None,
    advice: list[schema.Advice] | None = None,
    kind_edges: dict[str, set[str]] | None = None,
) -> schema.Result:
    custom_kinds = sorted(
        {entity.kind for entity in entities} - schema.WELL_KNOWN_ENTITY_KINDS
    )
    entities_by_id = {entity.id: entity for entity in entities}
    child_kinds_by_parent: dict[str, set[str]] = {}
    for entity in entities:
        for child_id in entity.child_ids:
            child = entities_by_id.get(child_id)
            if child is not None and entity.kind in custom_kinds:
                child_kinds_by_parent.setdefault(entity.kind, set()).add(child.kind)
        for parent_id in entity.parent_ids:
            parent = entities_by_id.get(parent_id)
            if parent is not None and parent.kind in custom_kinds:
                child_kinds_by_parent.setdefault(parent.kind, set()).add(entity.kind)
    for parent_kind, child_kinds in (kind_edges or {}).items():
        child_kinds_by_parent.setdefault(parent_kind, set()).update(child_kinds)
    return schema.Result(
        kind=schema.ResultKind.PERFORMANCE,
        status=schema.ResultStatus.OK,
        producer="test",
        entities=entities,
        entity_kinds=[
            schema.EntityKind(
                id=kind,
                child_kinds=sorted(child_kinds_by_parent.get(kind, set())),
            )
            for kind in custom_kinds
        ],
        breakdowns=breakdowns or [],
        checks=checks or [],
        advice=advice or [],
    )


def _output(result: schema.Result) -> dict[str, object]:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "timestamp": "2026-07-24T12:00:00Z",
        "tool": {"name": "MLIA", "version": "1.0.0"},
        "target": {
            "profile_name": "test",
            "target_type": "test",
            "components": ["test"],
            "configuration": {},
        },
        "model": {
            "name": "model.tflite",
            "format": "tflite",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "context": {},
        "backends": [{"name": "test", "version": "1.0.0"}],
        "results": [result.to_dict()],
    }


RULE = (CollapseRule("drop", "tag", ("yes",)),)


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        (
            [_entity("drop", kind="drop", attributes={"tag": "yes"}), _entity("keep")],
            {"keep": ([], [])},
        ),
        (
            [
                _entity("root", children=["drop"]),
                _entity(
                    "drop", kind="drop", children=["leaf"], attributes={"tag": "yes"}
                ),
                _entity("leaf"),
            ],
            {"root": ([], ["leaf"]), "leaf": (["root"], [])},
        ),
        (
            [
                _entity("root", children=["drop-a"]),
                _entity(
                    "drop-a",
                    kind="drop",
                    children=["drop-b"],
                    attributes={"tag": "yes"},
                ),
                _entity(
                    "drop-b", kind="drop", children=["leaf"], attributes={"tag": "yes"}
                ),
                _entity("leaf"),
            ],
            {"root": ([], ["leaf"]), "leaf": (["root"], [])},
        ),
        (
            [
                _entity("root", children=["drop"]),
                _entity(
                    "drop",
                    kind="drop",
                    children=["left", "right"],
                    attributes={"tag": "yes"},
                ),
                _entity("left"),
                _entity("right"),
            ],
            {
                "root": ([], ["left", "right"]),
                "left": (["root"], []),
                "right": (["root"], []),
            },
        ),
        (
            [
                _entity("left", children=["drop"]),
                _entity("right", children=["drop"]),
                _entity(
                    "drop", kind="drop", children=["leaf"], attributes={"tag": "yes"}
                ),
                _entity("leaf"),
            ],
            {
                "left": ([], ["leaf"]),
                "right": ([], ["leaf"]),
                "leaf": (["left", "right"], []),
            },
        ),
        (
            [
                _entity("root", children=["drop"]),
                _entity("drop", kind="drop", attributes={"tag": "yes"}),
            ],
            {"root": ([], [])},
        ),
    ],
)
def test_collapse_contracts_root_intermediate_consecutive_multi_parent_and_terminal(
    entities: list[schema.Entity], expected: dict[str, tuple[list[str], list[str]]]
) -> None:
    collapsed = collapse_entities(
        _result(entities, kind_edges={"node": {"node"}}), RULE
    )

    assert {
        entity.id: (entity.parent_ids, entity.child_ids)
        for entity in collapsed.entities
    } == expected


def test_parent_and_child_declarations_are_normalized_without_reciprocals() -> None:
    result = _result(
        [
            _entity("root", children=["drop"]),
            _entity(
                "drop",
                kind="drop",
                parents=[],
                children=["leaf"],
                attributes={"tag": "yes"},
            ),
            _entity("leaf", parents=[]),
        ]
    )

    result = replace(
        result,
        entity_kinds=[
            replace(kind, child_kinds=[*kind.child_kinds, "node"])
            if kind.id == "node"
            else kind
            for kind in result.entity_kinds
        ],
    )
    collapsed = collapse_entities(result, RULE)

    assert collapsed.entities[0].child_ids == ["leaf"]
    assert collapsed.entities[1].parent_ids == ["root"]


def test_matching_is_generic_raw_case_sensitive_and_string_only() -> None:
    rules = (
        CollapseRule("arbitrary", "value", ("A/*", "other")),
        CollapseRule("second", "name", ("match",)),
    )
    result = _result(
        [
            _entity("matched", kind="arbitrary", attributes={"value": "A/path"}),
            _entity("wrong-case", kind="arbitrary", attributes={"value": "a/path"}),
            _entity("backslash", kind="arbitrary", attributes={"value": r"A\path"}),
            _entity("missing", kind="arbitrary"),
            _entity("non-string", kind="arbitrary", attributes={"value": 7}),
            _entity("other-rule", kind="second", attributes={"name": "match"}),
        ]
    )

    collapsed = collapse_entities(result, rules)

    assert [entity.id for entity in collapsed.entities] == [
        "wrong-case",
        "backslash",
        "missing",
        "non-string",
    ]


def test_target_records_and_advice_references_are_cleaned_up(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = _result(
        [_entity("drop", kind="drop", attributes={"tag": "yes"}), _entity("keep")],
        breakdowns=[
            schema.Breakdown(entity_id="drop", metrics=[]),
            schema.Breakdown(entity_id="keep", metrics=[]),
        ],
        checks=[
            schema.Check(
                id="drop-check", status=schema.CheckStatus.PASS, entity_id="drop"
            ),
            schema.Check(
                id="keep-check", status=schema.CheckStatus.PASS, entity_id="keep"
            ),
        ],
        advice=[
            schema.Advice(
                id="drop-advice",
                category=schema.AdviceCategory.PERFORMANCE,
                severity=schema.AdviceSeverity.INFO,
                message="drop",
                affected_entity_ids=["drop"],
            ),
            schema.Advice(
                id="mixed",
                category=schema.AdviceCategory.PERFORMANCE,
                severity=schema.AdviceSeverity.INFO,
                message="mixed",
                affected_entity_ids=["drop", "keep"],
            ),
        ],
    )

    with caplog.at_level(logging.WARNING):
        collapsed = collapse_entities(result, RULE)

    assert [item.entity_id for item in collapsed.breakdowns] == ["keep"]
    assert [item.entity_id for item in collapsed.checks] == ["keep"]
    assert [item.id for item in collapsed.advice] == ["mixed"]
    assert collapsed.advice[0].affected_entity_ids == ["keep"]
    assert "Discarding 1 authoritative breakdown" in caplog.text
    assert "drop" in caplog.text


def test_invalid_original_graph_is_rejected_before_collapse() -> None:
    result = _result(
        [_entity("drop", kind="drop", children=["missing"], attributes={"tag": "yes"})]
    )

    with pytest.raises(ValueError, match="does not resolve"):
        collapse_entities(result, RULE)


def _metric(value: int) -> schema.Metric:
    return schema.Metric(name="cycles", value=value, unit="cycles")


def test_postprocessor_processes_every_standardized_result() -> None:
    first = _result(
        [
            _entity("drop-first", kind="drop", attributes={"tag": "yes"}),
            _entity("keep-first"),
        ]
    )
    second = _result(
        [
            _entity("drop-second", kind="drop", attributes={"tag": "yes"}),
            _entity("keep-second"),
        ]
    )
    output = _output(first)
    output["results"] = [first.to_dict(), second.to_dict()]
    settings = ApplicationSettings(filtering=FilteringSettings(collapse=RULE))

    processed = postprocess_standardized_output(output, settings)

    assert [
        [entity["id"] for entity in result["entities"]]
        for result in processed["results"]
    ] == [["keep-first"], ["keep-second"]]


def test_collapse_changes_residual_attribution_before_consumers_receive_output() -> (
    None
):
    result = _result(
        [
            _entity("target-stack", kind="code_stack", children=["A"]),
            _entity(
                "generated-stack",
                kind="code_stack",
                children=["G"],
                attributes={"file": "vendor/generated.py"},
            ),
            _entity("measured-chain", kind="chain", children=["A", "G"]),
            _entity("A", kind="source_operator"),
            _entity("G", kind="source_operator"),
        ],
        [schema.Breakdown(entity_id="measured-chain", metrics=[_metric(10)])],
    )
    assert not any(
        item.entity_id == "target-stack"
        for item in project_entity_breakdowns(result).breakdowns
    )

    settings = ApplicationSettings(
        filtering=FilteringSettings(
            collapse=(CollapseRule("code_stack", "file", ("vendor/*",)),)
        )
    )
    output = _output(result)

    processed = postprocess_standardized_output(output, settings)
    processed_result = processed["results"][0]

    assert {entity["id"] for entity in processed_result["entities"]} == {
        "target-stack",
        "measured-chain",
        "A",
        "G",
    }
    assert any(
        item["entity_id"] == "target-stack" and item["metrics"][0]["value"] == 10
        for item in processed_result["breakdowns"]
    )


def test_postprocessor_preserves_exact_result_dictionary_for_semantic_noop() -> None:
    """Explicit empty optional fields survive validation-only processing exactly."""
    raw_result = {
        "kind": "performance",
        "status": "ok",
        "producer": "test",
        "warnings": [],
        "errors": [],
        "metrics": [{"name": "summary", "value": 1, "unit": "", "qualifiers": {}}],
        "breakdowns": [{"entity_id": "keep", "metrics": [], "qualifiers": {}}],
        "checks": [
            {"id": "check", "status": "pass", "entity_id": "keep", "details": {}}
        ],
        "entities": [
            {
                "id": "keep",
                "kind": "node",
                "name": "keep",
                "parent_ids": [],
                "child_ids": [],
                "attributes": {},
                "stack_trace": "",
            }
        ],
        "entity_kinds": [{"id": "node", "parent_kinds": [], "child_kinds": []}],
        "advice": [
            {
                "id": "advice",
                "category": "performance",
                "severity": "info",
                "message": "no-op",
                "affected_entity_ids": [],
                "details": {},
            }
        ],
    }
    output = _output(_result([]))
    output["results"] = [raw_result]

    processed = postprocess_standardized_output(
        output,
        ApplicationSettings(filtering=FilteringSettings(collapse=())),
    )

    assert processed == output
    assert processed is not output
    assert processed["results"][0] is not raw_result


def test_postprocessor_translates_original_graph_failure() -> None:
    output = _output(_result([_entity("root", children=["missing"])]))

    with pytest.raises(SchemaValidationError, match="does not resolve"):
        postprocess_standardized_output(output, ApplicationSettings())


def test_postprocessor_validates_results_without_entities() -> None:
    output = _output(_result([]))
    output["results"] = [{}]

    with pytest.raises(SchemaValidationError, match="result 0.*kind"):
        postprocess_standardized_output(output, ApplicationSettings())


def test_postprocessor_validates_original_non_graph_references() -> None:
    output = _output(
        _result(
            [_entity("keep")],
            [schema.Breakdown(entity_id="missing", metrics=[])],
        )
    )

    with pytest.raises(SchemaValidationError, match="does not resolve"):
        postprocess_standardized_output(output, ApplicationSettings())


def test_postprocessor_validates_final_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_result = _result([_entity("keep")])
    invalid_result = _result(
        [_entity("keep")],
        [schema.Breakdown(entity_id="missing", metrics=[])],
    )
    monkeypatch.setattr(
        "mlia.core.output_postprocessing.project_entity_breakdowns",
        lambda _result: invalid_result,
    )

    with pytest.raises(SchemaValidationError, match="does not resolve"):
        postprocess_standardized_output(_output(valid_result), ApplicationSettings())


def test_postprocessor_derives_code_lines_after_configured_collapse() -> None:
    """Source lines should contain only retained stacks and receive projection."""
    result = _result(
        [
            _entity(
                "retained-stack",
                kind="code_stack",
                children=["A"],
                attributes={"file": "src/model.py", "line": 12},
            ),
            _entity(
                "collapsed-stack",
                kind="code_stack",
                children=["G"],
                attributes={"file": "vendor/generated.py", "line": 7},
            ),
            _entity("measured-chain", kind="chain", children=["A", "G"]),
            _entity("A", kind="source_operator"),
            _entity("G", kind="source_operator"),
        ],
        [schema.Breakdown(entity_id="measured-chain", metrics=[_metric(10)])],
    )
    settings = ApplicationSettings(
        filtering=FilteringSettings(
            collapse=(CollapseRule("code_stack", "file", ("vendor/*",)),)
        )
    )

    processed = postprocess_standardized_output(_output(result), settings)
    processed_result = processed["results"][0]
    lines = [
        entity
        for entity in processed_result["entities"]
        if entity["kind"] == schema.ENTITY_KIND_CODE_LINE
    ]

    assert len(lines) == 1
    assert lines[0]["attributes"] == {"file": "src/model.py", "line": 12}
    assert lines[0]["child_ids"] == ["retained-stack"]
    retained = next(
        entity
        for entity in processed_result["entities"]
        if entity["id"] == "retained-stack"
    )
    assert lines[0]["id"] in retained["parent_ids"]
    assert any(
        breakdown["entity_id"] == lines[0]["id"]
        and breakdown["metrics"][0]["value"] == 10
        for breakdown in processed_result["breakdowns"]
    )
    assert not any(
        entity["attributes"].get("file") == "vendor/generated.py" for entity in lines
    )


def test_collapse_retains_intermediate_for_undeclared_shortcut_kind() -> None:
    """Contraction must not invent an undeclared entity-kind relationship."""
    result = _result(
        [
            _entity("line", kind="source_line_view", children=["stack"]),
            _entity(
                "stack",
                kind="code_stack",
                children=["operator"],
                attributes={"file": "vendor/framework.py"},
            ),
            _entity("operator", kind="source_operator"),
        ]
    )

    collapsed = collapse_entities(
        result,
        (CollapseRule("code_stack", "file", ("vendor/*",)),),
    )

    assert [entity.id for entity in collapsed.entities] == [
        "line",
        "stack",
        "operator",
    ]
