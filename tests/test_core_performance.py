# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for target-neutral measured performance serialization."""

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.core import output_schema as schema
from mlia.core.output_validation import validate_standardized_output
from mlia.core.performance import MeasuredPerformanceMetrics, PerformanceObservations


def test_measured_output_hashes_model_without_reading_all_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model metadata uses streaming hashing and retains the correct digest."""
    payload = b"model artifact"
    model = tmp_path / "model.etrecord"
    model.write_bytes(payload)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        MagicMock(side_effect=AssertionError("Model hashing must stream the file")),
    )

    output = MeasuredPerformanceMetrics(
        measured_observations=[PerformanceObservations()]
    ).to_standardized_output(
        model,
        target=schema.Target(
            profile_name="example",
            target_type="example",
            components=[
                schema.Component(type=schema.ComponentType.CPU, family="example")
            ],
            configuration={},
        ),
        backend=schema.Backend(
            id="profiling", name="Profiling", version="1.0", configuration={}
        ),
    )

    assert output["model"]["hash"] == hashlib.sha256(payload).hexdigest()
    assert output["model"]["size_bytes"] == len(payload)
    validate_standardized_output(output)


@pytest.mark.parametrize(
    "measured_configuration, expected_configuration",
    [
        ({}, {"profiling_kind": "default", "keep": "value"}),
        (
            {"etdump": "capture.etdump"},
            {
                "profiling_kind": "default",
                "keep": "value",
                "etdump": "capture.etdump",
            },
        ),
        (
            {"profiling_kind": "standard-etdump"},
            {"profiling_kind": "standard-etdump", "keep": "value"},
        ),
    ],
    ids=["empty", "additional-fields", "measured-overrides-default"],
)
def test_measured_output_merges_backend_configuration(
    tmp_path: Path,
    measured_configuration: dict[str, Any],
    expected_configuration: dict[str, Any],
) -> None:
    """Measured configuration augments backend metadata without mutating inputs."""
    model = tmp_path / "model.etrecord"
    model.write_bytes(b"model artifact")
    backend = schema.Backend(
        id="profiling",
        name="Profiling",
        version="1.0",
        configuration={"profiling_kind": "default", "keep": "value"},
        impl={"provider": "example"},
    )
    metrics = MeasuredPerformanceMetrics(
        measured_observations=[PerformanceObservations()],
        backend_configuration=dict(measured_configuration),
    )

    output = metrics.to_standardized_output(
        model,
        target=schema.Target(
            profile_name="example",
            target_type="example",
            components=[
                schema.Component(type=schema.ComponentType.CPU, family="example")
            ],
            configuration={},
        ),
        backend=backend,
    )

    assert output["backends"] == [
        {**backend.to_dict(), "configuration": expected_configuration}
    ]
    assert output["results"][0]["producer"] == backend.id
    assert backend.configuration == {"profiling_kind": "default", "keep": "value"}
    assert metrics.backend_configuration == measured_configuration
    validate_standardized_output(output)
