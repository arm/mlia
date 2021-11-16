# Copyright 2021, Arm Ltd.
"""Tests for cli.commands module."""
# pylint: disable=no-self-use,too-many-arguments
from pathlib import Path
from typing import Any

import pytest
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.cli.common import ExecutionContext

from tests.test_cli_main import mock_performance_estimation


def test_performance_no_device(dummy_context: ExecutionContext) -> None:
    """Test that command should fail if no device provided."""
    with pytest.raises(Exception, match="Device is not provided"):
        performance(dummy_context, "some_model.tflite")


def test_performance_unknown_device(dummy_context: ExecutionContext) -> None:
    """Test that command should fail if unknown device passed."""
    with pytest.raises(Exception, match="Unsupported device: unknown"):
        performance(dummy_context, "some_model.tflite", device="unknown")


@pytest.mark.parametrize(
    "device, optimization_type, optimization_target, expected_error",
    [
        [
            "ethos-u55",
            None,
            "0.5",
            pytest.raises(Exception, match="Optimization type is not provided"),
        ],
        [
            "ethos-u65",
            "unknown",
            "16",
            pytest.raises(Exception, match="Unsupported optimization type: unknown"),
        ],
        [
            "ethos-u55",
            "pruning",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "ethos-u65",
            "clustering",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            None,
            "pruning",
            "0.5",
            pytest.raises(Exception, match="Device is not provided"),
        ],
        [
            "unknown",
            "clustering",
            "16",
            pytest.raises(Exception, match="Unsupported device: unknown"),
        ],
    ],
)
def test_opt_expected_parameters(
    dummy_context: ExecutionContext,
    device: str,
    optimization_type: str,
    optimization_target: str,
    expected_error: Any,
    test_models_path: Path,
) -> None:
    """Test that command should fail if no or unknown optimization type provided."""
    model = test_models_path / "simple_model.h5"

    with expected_error:
        optimization(
            dummy_context,
            str(model),
            device=device,
            optimization_type=optimization_type,
            optimization_target=optimization_target,
        )


@pytest.mark.parametrize(
    "device, optimization_type, optimization_target",
    [
        ["ethos-u55", "pruning", "0.5"],
        ["ethos-u65", "clustering", "32"],
        ["ethos-u55", "pruning,clustering", "0.5,32"],
    ],
)
def test_opt_valid_optimization_target(
    dummy_context: ExecutionContext,
    device: str,
    optimization_type: str,
    optimization_target: str,
    monkeypatch: Any,
    tmp_path: Path,
    test_models_path: Path,
) -> None:
    """Test that command should not fail with valid optimization targets."""
    model = test_models_path / "simple_model.h5"

    mock_performance_estimation(monkeypatch)

    optimization(
        dummy_context,
        str(model),
        device=device,
        optimization_type=optimization_type,
        optimization_target=optimization_target,
    )

    assert (tmp_path / "models/original_model.tflite").is_file()
    assert (tmp_path / "models/optimized_model.tflite").is_file()
