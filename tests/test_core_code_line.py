# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for core-derived source-line entities."""

from __future__ import annotations

import pytest

import mlia.core.output_schema as schema
from mlia.core.code_line import derive_code_line_entities


def _entity(
    entity_id: str,
    kind: str,
    *,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    attributes: dict | None = None,
) -> schema.Entity:
    return schema.Entity(
        id=entity_id,
        kind=kind,
        name=entity_id,
        parent_ids=parents or [],
        child_ids=children or [],
        attributes=attributes or {},
    )


def _result(entities: list[schema.Entity]) -> schema.Result:
    return schema.Result(
        kind=schema.ResultKind.PERFORMANCE,
        status=schema.ResultStatus.OK,
        producer="test",
        entities=entities,
    )


def test_derives_one_line_for_stacks_with_the_same_exact_location() -> None:
    """Equal source locations should share one deterministically ordered parent."""
    result = _result(
        [
            _entity(
                "stack-b",
                "code_stack",
                children=["source"],
                attributes={"file": "src/model.py", "line": 10},
            ),
            _entity(
                "stack-a",
                "code_stack",
                children=["source"],
                attributes={"file": "src/model.py", "line": 10},
            ),
            _entity("source", "source_operator"),
        ]
    )

    derived = derive_code_line_entities(result)
    line = derived.entities[-1]

    assert line.kind == schema.ENTITY_KIND_CODE_LINE
    assert line.id.startswith("code_line/model.py:10-")
    assert line.name == "model.py:10"
    assert line.child_ids == ["stack-a", "stack-b"]
    assert line.attributes == {"file": "src/model.py", "line": 10}
    assert derived.entities[0].parent_ids == [line.id]
    assert derived.entities[1].parent_ids == [line.id]


def test_materializes_normalized_stack_parents_before_adding_line_parent() -> None:
    """A caller declared only by child_ids must remain an explicit stack parent."""
    result = _result(
        [
            _entity("caller", "code_stack", children=["callee"]),
            _entity(
                "callee",
                "code_stack",
                attributes={"file": "src/model.py", "line": 20},
            ),
        ]
    )

    derived = derive_code_line_entities(result)
    line = derived.entities[-1]

    assert derived.entities[1].parent_ids == ["caller", line.id]


def test_skips_stacks_without_valid_well_known_location_attributes() -> None:
    """Malformed or absent source locations must not create guessed lines."""
    result = _result(
        [
            _entity("missing", "code_stack"),
            _entity("empty-file", "code_stack", attributes={"file": "", "line": 1}),
            _entity(
                "boolean-line",
                "code_stack",
                attributes={"file": "model.py", "line": True},
            ),
            _entity(
                "zero-line",
                "code_stack",
                attributes={"file": "model.py", "line": 0},
            ),
        ]
    )

    assert derive_code_line_entities(result) is result


def test_distinct_file_spellings_and_lines_create_distinct_entities() -> None:
    """Derivation should use the exact schema location rather than local mapping."""
    result = _result(
        [
            _entity(
                "first",
                "code_stack",
                attributes={"file": "src/model.py", "line": 1},
            ),
            _entity(
                "second",
                "code_stack",
                attributes={"file": "other/model.py", "line": 1},
            ),
            _entity(
                "third",
                "code_stack",
                attributes={"file": "src/model.py", "line": 2},
            ),
        ]
    )

    derived = derive_code_line_entities(result)
    lines = [
        entity
        for entity in derived.entities
        if entity.kind == schema.ENTITY_KIND_CODE_LINE
    ]

    assert len(lines) == 3
    assert len({entity.id for entity in lines}) == 3
    assert [entity.attributes for entity in lines] == [
        {"file": "other/model.py", "line": 1},
        {"file": "src/model.py", "line": 1},
        {"file": "src/model.py", "line": 2},
    ]


def test_rejects_backend_supplied_code_line_entities() -> None:
    """Core-derived entity ownership should not be ambiguous."""
    result = _result([_entity("code_line/backend", "code_line")])

    with pytest.raises(ValueError, match="must not define.*code_line/backend"):
        derive_code_line_entities(result)


def test_does_not_mutate_input() -> None:
    """Derivation should preserve every caller-owned entity container."""
    stack = _entity(
        "stack",
        "code_stack",
        parents=["caller"],
        attributes={"file": "src/model.py", "line": 7},
    )
    result = _result([_entity("caller", "code_stack", children=["stack"]), stack])
    before = result.to_dict()

    derived = derive_code_line_entities(result)

    assert result.to_dict() == before
    assert derived is not result
    assert derived.entities[1] is not stack
