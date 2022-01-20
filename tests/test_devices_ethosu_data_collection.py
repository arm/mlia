# Copyright 2022, Arm Ltd.
"""Tests for the data collection module for Ethos-U."""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mlia.core.context import Context
from mlia.core.data_collection import DataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.data_collection import EthosUOperatorCompatibility
from mlia.devices.ethosu.data_collection import EthosUOptimizationPerformance
from mlia.devices.ethosu.data_collection import EthosUPerformance
from mlia.devices.ethosu.performance import MemoryUsage
from mlia.devices.ethosu.performance import NPUCycles
from mlia.devices.ethosu.performance import OptimizationPerformanceMetrics
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.tools.vela_wrapper import Operators


@pytest.mark.parametrize(
    "collector, expected_name",
    [
        (
            EthosUOperatorCompatibility,
            "ethos_u_operator_compatibility",
        ),
        (
            EthosUPerformance,
            "ethos_u_performance",
        ),
        (
            EthosUOptimizationPerformance,
            "ethos_u_model_optimizations",
        ),
    ],
)
def test_collectors_metadata(
    collector: DataCollector,
    expected_name: str,
) -> None:
    """Test collectors metadata."""
    assert collector.name() == expected_name


def test_operator_compatibility_collector(
    test_models_path: Path, dummy_context: Context
) -> None:
    """Test operator compatibility data collector."""
    model = test_models_path / "simple_3_layers_model.tflite"
    device = EthosUConfiguration(target="U55-256")

    collector = EthosUOperatorCompatibility(model, device)
    collector.set_context(dummy_context)

    result = collector.collect_data()
    assert isinstance(result, Operators)


def test_performance_collector(
    monkeypatch: Any, test_models_path: Path, dummy_context: Context
) -> None:
    """Test performance data collector."""
    model = test_models_path / "simple_3_layers_model.tflite"
    device = EthosUConfiguration(target="U55-256")

    mock_performance_estimation(monkeypatch, device)

    collector = EthosUPerformance(model, device)
    collector.set_context(dummy_context)

    result = collector.collect_data()
    assert isinstance(result, PerformanceMetrics)


def test_optimization_performance_collector(
    monkeypatch: Any, test_models_path: Path, dummy_context: Context
) -> None:
    """Test optimization performance data collector."""
    model = test_models_path / "simple_model.h5"
    device = EthosUConfiguration(target="U55-256")

    mock_performance_estimation(monkeypatch, device)
    collector = EthosUOptimizationPerformance(
        model,
        device,
        [
            [
                {"optimization_type": "pruning", "optimization_target": 0.5},
            ]
        ],
    )
    collector.set_context(dummy_context)
    result = collector.collect_data()

    assert isinstance(result, OptimizationPerformanceMetrics)
    assert isinstance(result.original_perf_metrics, PerformanceMetrics)
    assert isinstance(result.optimizations_perf_metrics, list)
    assert len(result.optimizations_perf_metrics) == 1

    opt, metrics = result.optimizations_perf_metrics[0]
    assert opt == [OptimizationSettings("pruning", 0.5, None)]
    assert isinstance(metrics, PerformanceMetrics)

    collector_no_optimizations = EthosUOptimizationPerformance(
        model,
        device,
        [],
    )
    with pytest.raises(FunctionalityNotSupportedError):
        collector_no_optimizations.collect_data()

    tflite_model = test_models_path / "simple_3_layers_model.tflite"
    collector_tflite = EthosUOptimizationPerformance(
        tflite_model,
        device,
        [
            [
                {"optimization_type": "pruning", "optimization_target": 0.5},
            ]
        ],
    )
    collector_tflite.set_context(dummy_context)
    with pytest.raises(FunctionalityNotSupportedError):
        collector_tflite.collect_data()

    with pytest.raises(
        Exception, match="Optimization parameters expected to be a list"
    ):
        collector_bad_config = EthosUOptimizationPerformance(
            model, device, {"optimization_type": "pruning"}  # type: ignore
        )
        collector.set_context(dummy_context)
        collector_bad_config.collect_data()


def mock_performance_estimation(monkeypatch: Any, device: EthosUConfiguration) -> None:
    """Mock performance estimation."""
    metrics = PerformanceMetrics(
        device,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4, 5),
    )
    monkeypatch.setattr(
        "mlia.devices.ethosu.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )
