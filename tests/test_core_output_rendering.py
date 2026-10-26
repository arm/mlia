# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standardized output rendering."""

from __future__ import annotations

from typing import Any

import pytest

from mlia.core.output_rendering import _entities_table, standardized_output_to_text


def test_standardized_output_text_resolves_entity_references() -> None:
    """Breakdowns, checks, and advice should render referenced entity metadata."""
    output = {
        "target": {
            "profile_name": "test-profile",
            "target_type": "test-target",
            "components": [{"type": "npu", "family": "ethos-u"}],
            "configuration": {},
        },
        "model": {
            "name": "model.tflite",
            "format": "tflite",
            "hash": "a" * 64,
        },
        "backends": [
            {"id": "backend", "name": "Backend", "version": "1.0.0"},
        ],
        "results": [
            {
                "kind": "performance",
                "status": "ok",
                "producer": "backend",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "kind": "source_operator",
                        "name": "CONV_2D",
                        "placement": "NPU",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "metrics": [
                            {"name": "npu_cycles", "value": 1000, "unit": "cycles"}
                        ],
                    }
                ],
                "checks": [
                    {
                        "id": "operator_supported",
                        "status": "pass",
                        "entity_id": "source_operator/operator/0",
                    }
                ],
                "advice": [
                    {
                        "id": "0",
                        "category": "performance",
                        "severity": "info",
                        "message": "Review operator performance.",
                        "affected_entity_ids": ["source_operator/operator/0"],
                    }
                ],
            }
        ],
    }

    text = standardized_output_to_text(output)

    assert "CONV_2D" in text
    assert "operator/0" in text
    assert "npu_cycles=1000 cycles" in text
    assert "operator_supported" in text
    assert "Review operator performance." in text


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            {
                "reasons": [
                    {"description": "Torch operation not lowered", "detail": ""},
                    {"description": "TOSA rank violation", "detail": ""},
                ]
            },
            "Torch operation not lowered; TOSA rank violation",
        ),
        (
            {
                "reasons": [
                    {"description": "Unsupported shape", "detail": "Rank must be 4"}
                ]
            },
            "Unsupported shape: Rank must be 4",
        ),
        (
            {"reasons": [{"description": "Unsupported datatype", "detail": ""}]},
            "Unsupported datatype",
        ),
        ({"reasons": ["Rejected", "Rejected"]}, "Rejected"),
        ({"reasons": []}, "-"),
        (None, "-"),
        ({"reasons": [None, {}]}, "-"),
    ],
)
def test_entity_table_displays_backend_reasons(details: object, expected: str) -> None:
    """All targets render reasons through the same entity-linked checks."""
    entity = {
        "id": "op/0",
        "name": "custom",
        "kind": "source_operator",
        "placement": "CPU",
    }
    output = {
        "results": [
            {
                "entities": [entity],
                "checks": [
                    {"id": "support", "entity_id": "op/0", "details": details},
                    {
                        "id": "other",
                        "entity_id": "op/1",
                        "details": {
                            "reasons": [{"description": "Unrelated rejection"}]
                        },
                    },
                    {
                        "id": "global",
                        "details": {"reasons": [{"description": "Whole-model error"}]},
                    },
                ],
            }
        ]
    }
    text = standardized_output_to_text(output)
    assert ("Reason" in text) == (expected != "-")
    assert "Kind" in text
    assert "source_operator" in text
    assert expected in text
    assert "Unrelated rejection" not in text
    assert "Whole-model error" not in text
    assert entity["kind"] == "source_operator"


def test_mixed_entity_table_keeps_reason_and_kind() -> None:
    """One rejection exposes Reason, while other operators retain empty cells."""
    text = _entities_table(
        [
            {
                "id": "0",
                "name": "supported",
                "kind": "source_operator",
                "placement": "NPU",
            },
            {
                "id": "1",
                "name": "fallback",
                "kind": "source_operator",
                "placement": "CPU",
            },
        ],
        [
            {
                "entity_id": "1",
                "details": {"reasons": [{"description": "Rejected", "detail": ""}]},
            }
        ],
    )
    assert text is not None
    assert "Reason" in text
    assert "Kind" in text
    assert "Rejected" in text
    supported_row = next(line for line in text.splitlines() if "supported" in line)
    assert "-" in supported_row


def test_long_entity_reason_wraps_without_losing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing Rich table wraps diagnostic text to terminal width."""
    monkeypatch.setattr("mlia.utils.console._output_width", lambda: 100)
    text = _entities_table(
        [{"id": "0", "name": "op", "kind": "source_operator", "placement": "CPU"}],
        [
            {
                "entity_id": "0",
                "details": {
                    "reasons": [{"description": "explanation " * 40, "detail": ""}]
                },
            }
        ],
    )
    assert text is not None
    assert text.count("explanation") == 40
    assert all(len(line) <= 100 for line in text.splitlines())


@pytest.mark.parametrize(
    "reason",
    [
        "dtype [int8] is unsupported",
        "unexpected closing tag [/int8]",
        "[bold]literal backend diagnostic[/bold]",
        r"dtype \[int8] is unsupported",
    ],
)
def test_entity_reasons_are_literal_text(reason: str) -> None:
    """Backend diagnostics must not be interpreted as Rich markup."""
    output: dict[str, Any] = {
        "results": [
            {
                "entities": [{"id": "0", "name": "op", "placement": "CPU"}],
                "checks": [
                    {
                        "entity_id": "0",
                        "details": {
                            "reasons": [{"description": reason, "detail": reason}]
                        },
                    }
                ],
            }
        ]
    }
    text = standardized_output_to_text(output)
    assert text.count(reason) == 2
    assert output["results"][0]["checks"][0]["details"]["reasons"][0] == {
        "description": reason,
        "detail": reason,
    }
