# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/performance."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compiler import optimize_model
from mlia.backend.vela.performance import estimate_performance
from mlia.backend.vela.performance import PerformanceMetrics
from mlia.devices.ethosu.config import EthosUConfiguration


def test_estimate_performance(test_tflite_model: Path) -> None:
    """Test getting performance estimations."""
    device = EthosUConfiguration("ethos-u55-256")
    perf_metrics = estimate_performance(test_tflite_model, device.compiler_options)

    assert isinstance(perf_metrics, PerformanceMetrics)


def test_estimate_performance_already_optimized(
    tmp_path: Path, test_tflite_model: Path
) -> None:
    """Test that performance estimation should fail for already optimized model."""
    device = EthosUConfiguration("ethos-u55-256")

    optimized_model_path = tmp_path / "optimized_model.tflite"

    optimize_model(test_tflite_model, device.compiler_options, optimized_model_path)

    with pytest.raises(
        Exception, match="Unable to estimate performance for the given optimized model"
    ):
        estimate_performance(optimized_model_path, device.compiler_options)


def test_read_invalid_model(test_tflite_invalid_model: Path) -> None:
    """Test that reading invalid model should fail with exception."""
    with pytest.raises(
        Exception, match=f"Unable to read model {test_tflite_invalid_model}"
    ):
        device = EthosUConfiguration("ethos-u55-256")
        estimate_performance(test_tflite_invalid_model, device.compiler_options)


def test_compile_invalid_model(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that if model could not be compiled then correct exception raised."""
    mock_compiler = MagicMock()
    mock_compiler.side_effect = Exception("Bad model!")

    monkeypatch.setattr("mlia.backend.vela.compiler.compiler_driver", mock_compiler)

    model_path = tmp_path / "optimized_model.tflite"
    with pytest.raises(
        Exception, match="Model could not be optimized with Vela compiler"
    ):
        device = EthosUConfiguration("ethos-u55-256")
        optimize_model(test_tflite_model, device.compiler_options, model_path)

    assert not model_path.exists()
