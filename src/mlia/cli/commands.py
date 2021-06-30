"""Cli commands module."""
from typing import Any

from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.operators import supported_operators
from mlia.performance import collect_performance_metrics
from mlia.reporters import report_performance_estimation
from mlia.reporters import report_supported_operators


def operators(model: str, **device_args: Any) -> None:
    """Print the model's operator list."""
    tflite_model, device = TFLiteModel(model), get_device(**device_args)
    ops = supported_operators(tflite_model, device)
    report_supported_operators(ops)


def performance(model: str, **device_args: Any) -> None:
    """Print model's performance stats."""
    tflite_model, device = TFLiteModel(model), get_device(**device_args)
    perf_metrics = collect_performance_metrics(tflite_model, device)
    report_performance_estimation(perf_metrics)


def get_device(**kwargs: Any) -> EthosUConfiguration:
    """Get device configuration."""
    device = kwargs.pop("device", None)
    if not device:
        raise Exception("Device is not provided")

    if device == "ethos-u55":
        return EthosU55(**kwargs)
    elif device == "ethos-u65":
        return EthosU65(**kwargs)

    raise Exception(f"Unsupported device {device}")
