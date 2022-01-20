# Copyright 2021, Arm Ltd.
"""Tests for cli.commands module."""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mlia.cli.commands import operators
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.core.context import ExecutionContext
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.performance import MemoryUsage
from mlia.devices.ethosu.performance import NPUCycles
from mlia.devices.ethosu.performance import PerformanceMetrics


def test_operators_expected_parameters(dummy_context: ExecutionContext) -> None:
    """Test operators command wrong parameters."""
    with pytest.raises(Exception, match="Model is not provided"):
        operators(dummy_context, "U55-256")


def test_performance_unknown_target(
    dummy_context: ExecutionContext, test_models_path: Path
) -> None:
    """Test that command should fail if unknown target passed."""
    model = test_models_path / "simple_3_layers_model.tflite"
    with pytest.raises(Exception, match="Unable to find target profile unknown"):
        performance(dummy_context, model=str(model), target="unknown")


@pytest.mark.parametrize(
    "target, optimization_type, optimization_target, expected_error",
    [
        [
            "U55-256",
            None,
            "0.5",
            pytest.raises(Exception, match="Optimization type is not provided"),
        ],
        [
            "U65-512",
            "unknown",
            "16",
            pytest.raises(Exception, match="Unsupported optimization type: unknown"),
        ],
        [
            "U55-256",
            "pruning",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "U65-512",
            "clustering",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "unknown",
            "clustering",
            "16",
            pytest.raises(Exception, match="Unable to find target profile unknown"),
        ],
    ],
)
def test_opt_expected_parameters(
    dummy_context: ExecutionContext,
    target: str,
    monkeypatch: Any,
    optimization_type: str,
    optimization_target: str,
    expected_error: Any,
    test_models_path: Path,
) -> None:
    """Test that command should fail if no or unknown optimization type provided."""
    model = test_models_path / "simple_model.h5"

    mock_performance_estimation(monkeypatch)

    with expected_error:
        optimization(
            ctx=dummy_context,
            target=target,
            model=str(model),
            optimization_type=optimization_type,
            optimization_target=optimization_target,
        )


@pytest.mark.parametrize(
    "target, optimization_type, optimization_target",
    [
        ["U55-256", "pruning", "0.5"],
        ["U65-512", "clustering", "32"],
        ["U55-256", "pruning,clustering", "0.5,32"],
    ],
)
def test_opt_valid_optimization_target(
    target: str,
    dummy_context: ExecutionContext,
    optimization_type: str,
    optimization_target: str,
    monkeypatch: Any,
    test_models_path: Path,
) -> None:
    """Test that command should not fail with valid optimization targets."""
    model = test_models_path / "simple_model.h5"

    mock_performance_estimation(monkeypatch)

    optimization(
        ctx=dummy_context,
        target=target,
        model=str(model),
        optimization_type=optimization_type,
        optimization_target=optimization_target,
    )


def mock_performance_estimation(monkeypatch: Any) -> None:
    """Mock performance estimation."""
    metrics = PerformanceMetrics(
        EthosUConfiguration("U55-256"),
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4, 5),
    )
    monkeypatch.setattr(
        "mlia.devices.ethosu.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )
