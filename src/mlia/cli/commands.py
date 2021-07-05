# Copyright 2021, Arm Ltd.
"""Cli commands module."""
import sys
from typing import Any

from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.operators import supported_operators
from mlia.performance import collect_performance_metrics
from mlia.reporters import report


def operators(
    model: str,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **device_args: Any,
) -> None:
    """Print the model's operator list."""
    tflite_model, device = TFLiteModel(model), get_device(**device_args)
    ops = supported_operators(tflite_model, device)

    report(ops, fmt=output_format, output=output)


def performance(
    model: str,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **device_args: Any,
) -> None:
    """Print model's performance stats."""
    tflite_model, device = TFLiteModel(model), get_device(**device_args)
    perf_metrics = collect_performance_metrics(tflite_model, device)

    report(perf_metrics, fmt=output_format, output=output)


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
