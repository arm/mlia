# Copyright 2021, Arm Ltd.
"""Tests for cli.commands module."""
from typing import Any
from typing import Union

import pytest
from mlia.cli.commands import estimate_optimized_performance
from mlia.cli.commands import model_optimization
from mlia.cli.commands import performance
from mlia.utils.general import save_keras_model

from tests.test_cli_main import mock_performance_estimation
from tests.utils.generate_keras_model import generate_keras_model


def test_command_no_device() -> None:
    """Test that command should fail if no device provided."""
    with pytest.raises(Exception, match="Device is not provided"):
        performance("some_model.tflite")


def test_command_unknown_device() -> None:
    """Test that command should fail if unknown device passed."""
    with pytest.raises(Exception, match="Unsupported device: unknown"):
        performance("some_model.tflite", device="unknown")


@pytest.mark.parametrize(
    "optimization_type, optimization_target, expected_error",
    [
        [
            None,
            0.5,
            pytest.raises(Exception, match="Optimization type is not provided"),
        ],
        [
            "unknown",
            0.5,
            pytest.raises(Exception, match="Unsupported optimization type: unknown"),
        ],
        [
            "pruning",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "clustering",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
    ],
)
def test_command_expected_parameters(
    optimization_type: str, optimization_target: Union[int, float], expected_error: Any
) -> None:
    """Test that command should fail if no or unknown optimization type provided."""
    model = generate_keras_model()
    model_path = save_keras_model(model)
    with expected_error:
        model_optimization(
            model_path,
            optimization_type=optimization_type,
            optimization_target=optimization_target,
        )


@pytest.mark.parametrize(
    "optimization_type, optimization_target",
    [
        ["pruning", -1.0],
        ["pruning", 1.1],
        ["pruning", 2.0],
        ["clustering", -42],
        ["clustering", -1],
        ["clustering", 0],
        ["clustering", 1],
        ["clustering", 2],
        ["clustering", 0],
        ["clustering", 4.20],
    ],
)
def test_command_invalid_optimization_target(
    optimization_type: str, optimization_target: Union[int, float]
) -> None:
    """Test that command should fail if optimization target out of bounds."""
    model = generate_keras_model()
    model_path = save_keras_model(model)
    with pytest.raises(Exception):
        model_optimization(
            model_path,
            optimization_type=optimization_type,
            optimization_target=optimization_target,
        )


@pytest.mark.parametrize(
    "optimization_type, optimization_target",
    [
        ["pruning", 0.5],
        ["clustering", 3],
    ],
)
def test_mopt_valid_optimization_target(
    optimization_type: str, optimization_target: Union[int, float]
) -> None:
    """Test that command should not fail with valid optimization targets."""
    model = generate_keras_model()
    model_path = save_keras_model(model)
    model_optimization(
        model_path,
        optimization_type=optimization_type,
        optimization_target=optimization_target,
    )


@pytest.mark.parametrize(
    "device, optimization_type, optimization_target",
    [
        ["ethos-u55", "pruning", 0.5],
        ["ethos-u65", "clustering", 32],
    ],
)
def test_eop_valid_optimization_target(
    device: str,
    optimization_type: str,
    optimization_target: Union[int, float],
    monkeypatch: Any,
) -> None:
    """Test that command should not fail with valid optimization targets."""
    model = generate_keras_model()
    model_path = save_keras_model(model)

    mock_performance_estimation(monkeypatch)

    estimate_optimized_performance(
        model_path,
        device=device,
        optimization_type=optimization_type,
        optimization_target=optimization_target,
    )
