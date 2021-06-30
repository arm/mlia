# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation tests."""
from pathlib import Path

import pytest
from ethosu.vela.errors import InputFileError
from mlia.config import EthosU55
from mlia.config import TFLiteModel
from mlia.exceptions import ConfigurationError
from mlia.performance import collect_performance_metrics
from mlia.performance import PerformanceMetrics


def test_collect_performance_metrics(test_models_path: Path) -> None:
    """Test collect_performance_metrics function."""
    # Test empty path/model
    with pytest.raises(InputFileError):
        performance_metrics = collect_performance_metrics(TFLiteModel(""), EthosU55())

    # Test non-existing path/model
    with pytest.raises(FileNotFoundError):
        performance_metrics = collect_performance_metrics(
            TFLiteModel("invalid_model.tflite"), EthosU55()
        )

    with pytest.raises(ConfigurationError):
        collect_performance_metrics("model.tflite", "ethos-u55")  # type: ignore

    with pytest.raises(ConfigurationError):
        model = TFLiteModel("model.tflite")
        collect_performance_metrics(model, "ethos-u55")  # type: ignore

    # Test valid path/model
    input_tflite = test_models_path / "simple_3_layers_model.tflite"
    performance_metrics = collect_performance_metrics(
        TFLiteModel(str(input_tflite)), EthosU55()
    )

    assert isinstance(performance_metrics, PerformanceMetrics)

    assert performance_metrics.npu_cycles >= 0
    assert performance_metrics.sram_access_cycles >= 0
    assert performance_metrics.dram_access_cycles >= 0
    assert performance_metrics.on_chip_flash_access_cycles >= 0
    assert performance_metrics.off_chip_flash_access_cycles >= 0
    assert performance_metrics.total_cycles >= 0

    assert performance_metrics.batch_inference_time >= 0
    assert performance_metrics.inferences_per_second >= 0
    assert performance_metrics.batch_size >= 1
