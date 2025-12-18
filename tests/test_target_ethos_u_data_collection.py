# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the data collection module for Ethos-U."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.corstone.performance import PerformanceMetrics as CorstonePerf
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import VelaCompatibilityResult
from mlia.backend.vela.performance import LayerwisePerfInfo
from mlia.backend.vela.performance import PerformanceMetrics as VelaPerf
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
from mlia.target.ethos_u.performance import CombinedPerformanceResult
from mlia.target.ethos_u.performance import CorstonePerformanceResult
from mlia.target.ethos_u.performance import MemoryUsage
from mlia.target.ethos_u.performance import NPUCycles
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.target.ethos_u.performance import VelaPerformanceResult


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

    try:
        result = collector.collect_data()
        # Should return VelaCompatibilityResult wrapper with standardized output
        assert isinstance(result, VelaCompatibilityResult)
        assert result.legacy_info is not None
        assert isinstance(result.legacy_info, Operators)
        assert result.standardized_output is not None
    except BackendUnavailableError:
        # If Vela is not available, the test should pass (expected behavior)
        pytest.skip("Vela backend not available, skipping operator compatibility test")


def test_performance_collector(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation(monkeypatch, target)

    collector = EthosUPerformance(test_tflite_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    # Without backends specified, collector returns PerformanceMetrics
    assert isinstance(result, PerformanceMetrics)


def test_performance_collector_with_vela(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector with Vela backend."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation_with_vela(monkeypatch, target)

    collector = EthosUPerformance(test_tflite_model, target, backends=["vela"])
    collector.set_context(sample_context)

    result = collector.collect_data()
    # With only Vela backend, collector returns VelaPerformanceResult
    assert isinstance(result, VelaPerformanceResult)
    assert result.standardized_output is not None
    assert isinstance(result.standardized_output, dict)


def test_performance_collector_with_corstone(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector with Corstone backend."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation_with_corstone(monkeypatch, target)

    collector = EthosUPerformance(test_tflite_model, target, backends=["corstone-310"])
    collector.set_context(sample_context)

    result = collector.collect_data()
    # With only Corstone backend, collector returns CorstonePerformanceResult
    assert isinstance(result, CorstonePerformanceResult)
    assert result.standardized_output is not None
    assert isinstance(result.standardized_output, dict)


def test_performance_collector_with_both_backends(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector with both Vela and Corstone backends."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation_with_both(monkeypatch, target)

    collector = EthosUPerformance(
        test_tflite_model, target, backends=["vela", "corstone-310"]
    )
    collector.set_context(sample_context)

    result = collector.collect_data()
    # With both backends, collector returns CombinedPerformanceResult
    assert isinstance(result, CombinedPerformanceResult)
    assert result.standardized_output is not None
    assert isinstance(result.standardized_output, dict)
    # Check that both backends are present
    assert len(result.standardized_output["backends"]) == 2
    # Check that both results are present
    assert len(result.standardized_output["results"]) == 2


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
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )


def mock_performance_estimation_with_vela(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation with Vela metrics."""
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )

    mock_estimator = MagicMock()
    mock_estimator.estimate.return_value = metrics
    mock_estimator.vela_perf_metrics = MagicMock(spec=VelaPerf)
    mock_estimator.vela_perf_metrics.to_standardized_output.return_value = {
        "schema_version": "1.0.0",
        "backends": [{"id": "vela"}],
        "results": [{"kind": "performance"}],
    }
    mock_estimator.vela_compiler_options = None

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=mock_estimator),
    )


def mock_performance_estimation_with_corstone(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation with Corstone metrics."""
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    metrics.corstone_metrics = MagicMock(spec=CorstonePerf)
    metrics.to_standardized_output = MagicMock(  # type: ignore[method-assign]
        return_value={
            "schema_version": "1.0.0",
            "backends": [{"id": "corstone-310"}],
            "results": [{"kind": "performance"}],
        }
    )

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )


def mock_performance_estimation_with_both(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation with both Vela and Corstone metrics."""
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    metrics.corstone_metrics = MagicMock(spec=CorstonePerf)
    metrics.to_standardized_output = MagicMock(  # type: ignore[method-assign]
        return_value={
            "schema_version": "1.0.0",
            "backends": [{"id": "corstone-310"}],
            "results": [{"kind": "performance", "producer": "corstone-310"}],
        }
    )

    mock_estimator = MagicMock()
    mock_estimator.estimate.return_value = metrics
    mock_estimator.vela_perf_metrics = MagicMock(spec=VelaPerf)
    mock_estimator.vela_perf_metrics.to_standardized_output.return_value = {
        "schema_version": "1.0.0",
        "backends": [{"id": "vela"}],
        "results": [{"kind": "performance", "producer": "vela"}],
    }
    mock_estimator.vela_compiler_options = None

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=mock_estimator),
    )
