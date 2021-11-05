# Copyright 2021, Arm Ltd.
"""Performance estimation tests."""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from ethosu.vela.errors import InputFileError
from mlia.config import Context
from mlia.config import EthosU55
from mlia.config import TFLiteModel
from mlia.exceptions import ConfigurationError
from mlia.metrics import PerformanceMetrics
from mlia.performance import collect_performance_metrics


def test_collect_performance_metrics(
    dummy_context: Context, test_models_path: Path, monkeypatch: Any
) -> None:
    """Test collect_performance_metrics function."""
    # Test empty path/model
    with pytest.raises(InputFileError):
        performance_metrics = collect_performance_metrics(
            TFLiteModel(""), EthosU55(), dummy_context
        )

    # Test non-existing path/model
    with pytest.raises(FileNotFoundError):
        performance_metrics = collect_performance_metrics(
            TFLiteModel("invalid_model.tflite"), EthosU55(), dummy_context
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
        TFLiteModel(str(input_tflite)), EthosU55(), dummy_context
    )

    assert isinstance(performance_metrics, PerformanceMetrics)


def mock_performance_estimation(monkeypatch: Any) -> None:
    """Mock performance estimation."""
    monkeypatch.setattr(
        "mlia.tools.vela_wrapper.estimate_performance",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr("mlia.tools.vela_wrapper.optimize_model", MagicMock())

    monkeypatch.setattr(
        "mlia.tools.aiet_wrapper.estimate_performance",
        MagicMock(return_value=MagicMock()),
    )
