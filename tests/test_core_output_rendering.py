# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standardized output rendering."""

from __future__ import annotations

from mlia.core.output_rendering import standardized_output_to_text


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
