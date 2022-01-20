# Copyright 2021, Arm Ltd.
"""Performance estimation tests."""
from pathlib import Path
from typing import Any
from typing import Union
from unittest.mock import MagicMock

import pandas as pd
import pytest
from mlia.core.context import Context
from mlia.core.errors import ConfigurationError
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.performance import collect_performance_metrics
from mlia.devices.ethosu.performance import MemorySizeType
from mlia.devices.ethosu.performance import MemoryUsage
from mlia.devices.ethosu.performance import NPUCycles
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.nn.tensorflow.config import TFLiteModel


def test_memory_usage_conversion() -> None:
    """Test MemoryUsage objects conversion."""
    memory_usage_in_kb = MemoryUsage(1, 2, 3, 4, 5, MemorySizeType.KILOBYTES)
    assert memory_usage_in_kb.in_kilobytes() == memory_usage_in_kb

    memory_usage_in_bytes = MemoryUsage(
        1 * 1024, 2 * 1024, 3 * 1024, 4 * 1024, 5 * 1024
    )
    assert memory_usage_in_bytes.in_kilobytes() == memory_usage_in_kb


def test_collect_performance_metrics(
    dummy_context: Context, test_models_path: Path, monkeypatch: Any
) -> None:
    """Test collect_performance_metrics function."""
    # Test empty path/model
    with pytest.raises(Exception, match="Model path is not provided"):
        performance_metrics = collect_performance_metrics(
            TFLiteModel(""), EthosUConfiguration(target="U55-256"), dummy_context
        )

    # Test non-existing path/model
    with pytest.raises(FileNotFoundError):
        performance_metrics = collect_performance_metrics(
            TFLiteModel("invalid_model.tflite"),
            EthosUConfiguration(target="U55-256"),
            dummy_context,
        )

    with pytest.raises(ConfigurationError):
        collect_performance_metrics(
            "model.tflite", "ethos-u55", dummy_context  # type: ignore
        )

    with pytest.raises(ConfigurationError):
        model = TFLiteModel("model.tflite")
        collect_performance_metrics(model, "ethos-u55", dummy_context)  # type: ignore

    # Test valid path/model
    mock_performance_estimation(monkeypatch)

    input_tflite = test_models_path / "simple_3_layers_model.tflite"
    performance_metrics = collect_performance_metrics(
        TFLiteModel(str(input_tflite)),
        EthosUConfiguration(target="U55-256"),
        dummy_context,
    )

    assert isinstance(performance_metrics, PerformanceMetrics)


def mock_performance_estimation(monkeypatch: Any) -> None:
    """Mock performance estimation."""
    monkeypatch.setattr(
        "mlia.tools.aiet_wrapper.estimate_performance",
        MagicMock(return_value=MagicMock()),
    )


@pytest.mark.parametrize(
    "metric, dataframe",
    [
        (
            NPUCycles(1, 2, 3, 4, 5, 6),
            pd.DataFrame.from_records(
                [[1, 2, 3, 4, 5, 6]],
                columns=[
                    "NPU active cycles",
                    "NPU idle cycles",
                    "NPU total cycles",
                    "NPU AXI0 RD data beat received",
                    "NPU AXI0 WR data beat written",
                    "NPU AXI1 RD data beat received",
                ],
            ),
        ),
        (
            MemoryUsage(1, 2, 3, 4, 5),
            pd.DataFrame.from_records(
                [[1, 2, 3, 4, 5]],
                columns=[
                    "SRAM used (bytes)",
                    "DRAM used (bytes)",
                    "Unknown memory used (bytes)",
                    "On chip flash used (bytes)",
                    "Off chip flash used (bytes)",
                ],
            ),
        ),
        (
            MemoryUsage(1024, 1024, 1024, 1024, 1024).in_kilobytes(),
            pd.DataFrame.from_records(
                [[1.0, 1.0, 1.0, 1.0, 1.0]],
                columns=[
                    "SRAM used (KiB)",
                    "DRAM used (KiB)",
                    "Unknown memory used (KiB)",
                    "On chip flash used (KiB)",
                    "Off chip flash used (KiB)",
                ],
            ),
        ),
        (
            PerformanceMetrics(
                EthosUConfiguration(target="U55-256"),
                NPUCycles(1, 2, 3, 4, 5, 6),
                MemoryUsage(1, 2, 3, 4, 5),
            ),
            pd.DataFrame.from_records(
                [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]],
                columns=[
                    "SRAM used (bytes)",
                    "DRAM used (bytes)",
                    "Unknown memory used (bytes)",
                    "On chip flash used (bytes)",
                    "Off chip flash used (bytes)",
                    "NPU active cycles",
                    "NPU idle cycles",
                    "NPU total cycles",
                    "NPU AXI0 RD data beat received",
                    "NPU AXI0 WR data beat written",
                    "NPU AXI1 RD data beat received",
                ],
            ),
        ),
        (
            PerformanceMetrics(
                EthosUConfiguration(target="U55-256"),
                NPUCycles(1, 2, 3, 4, 5, 6),
                MemoryUsage(1024, 2 * 1024, 3 * 1024, 4 * 1024, 5 * 1024),
            ).in_kilobytes(),
            pd.DataFrame.from_records(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 1, 2, 3, 4, 5, 6]],
                columns=[
                    "SRAM used (KiB)",
                    "DRAM used (KiB)",
                    "Unknown memory used (KiB)",
                    "On chip flash used (KiB)",
                    "Off chip flash used (KiB)",
                    "NPU active cycles",
                    "NPU idle cycles",
                    "NPU total cycles",
                    "NPU AXI0 RD data beat received",
                    "NPU AXI0 WR data beat written",
                    "NPU AXI1 RD data beat received",
                ],
            ),
        ),
    ],
)
def test_object_to_dataframe_conversion(
    metric: Union[MemoryUsage, NPUCycles, PerformanceMetrics], dataframe: pd.DataFrame
) -> None:
    """Test object to dataframe conversion."""
    assert metric.to_df().equals(dataframe)
