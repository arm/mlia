# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the data collection module for Ethos-U."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compat import Operators
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.data_collection import DataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.nn.select import OptimizationSettings
from mlia.target.common.optimization import add_common_optimization_params
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.data_collection import EthosUOperatorCompatibility
from mlia.target.ethos_u.data_collection import EthosUOptimizationPerformance
from mlia.target.ethos_u.data_collection import EthosUPerformance
from mlia.target.ethos_u.performance import MemoryUsage
from mlia.target.ethos_u.performance import NPUCycles
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics


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


def setup_optimization(optimizations: list) -> Context:
    """Set up optimization params for the context."""
    params: dict = {}
    add_common_optimization_params(
        params,
        {
            "optimization_targets": optimizations,
        },
    )

    context = ExecutionContext(config_parameters=params)
    return context


def test_operator_compatibility_collector(
    sample_context: Context, test_tflite_model: Path
) -> None:
    """Test operator compatibility data collector."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    collector = EthosUOperatorCompatibility(test_tflite_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    assert isinstance(result, Operators)


def test_performance_collector(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation(monkeypatch, target)

    collector = EthosUPerformance(test_tflite_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    assert isinstance(result, PerformanceMetrics)


def test_optimization_performance_collector(
    monkeypatch: pytest.MonkeyPatch,
    test_keras_model: Path,
    test_tflite_model: Path,
) -> None:
    """Test optimization performance data collector."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation(monkeypatch, target)

    context = setup_optimization(
        [
            {"optimization_type": "pruning", "optimization_target": 0.5},
        ],
    )
    collector = EthosUOptimizationPerformance(test_keras_model, target)
    collector.set_context(context)
    result = collector.collect_data()

    assert isinstance(result, OptimizationPerformanceMetrics)
    assert isinstance(result.original_perf_metrics, PerformanceMetrics)
    assert isinstance(result.optimizations_perf_metrics, list)
    assert len(result.optimizations_perf_metrics) == 1

    opt, metrics = result.optimizations_perf_metrics[0]
    assert opt == [OptimizationSettings("pruning", 0.5, None)]
    assert isinstance(metrics, PerformanceMetrics)

    context = ExecutionContext(
        config_parameters={"common_optimizations": {"optimizations": [[]]}}
    )

    collector_no_optimizations = EthosUOptimizationPerformance(test_keras_model, target)
    collector_no_optimizations.set_context(context)
    with pytest.raises(FunctionalityNotSupportedError):
        collector_no_optimizations.collect_data()

    context = setup_optimization(
        [
            {"optimization_type": "pruning", "optimization_target": 0.5},
        ],
    )

    collector_tflite = EthosUOptimizationPerformance(test_tflite_model, target)
    collector_tflite.set_context(context)
    with pytest.raises(FunctionalityNotSupportedError):
        collector_tflite.collect_data()

    with pytest.raises(
        Exception, match="Optimization parameters expected to be a list"
    ):
        context = ExecutionContext(
            config_parameters={
                "common_optimizations": {
                    "optimizations": [{"optimization_type": "pruning"}]
                }
            }
        )

        collector_bad_config = EthosUOptimizationPerformance(test_keras_model, target)
        collector_bad_config.set_context(context)
        collector_bad_config.collect_data()


def mock_performance_estimation(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation."""
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4, 5),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )
