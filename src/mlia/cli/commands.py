# Copyright 2021, Arm Ltd.
"""Cli commands module."""
import logging
import sys
from typing import Any
from typing import Optional

import tensorflow as tf
from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.operators import supported_operators
from mlia.optimizations.clustering import Clusterer
from mlia.optimizations.clustering import ClusteringConfiguration
from mlia.optimizations.common import Optimizer
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.pruning import PruningConfiguration
from mlia.performance import collect_performance_metrics
from mlia.reporters import report
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import save_keras_model
from mlia.utils.general import save_tflite_model

LOGGER = logging.getLogger("mlia.cli")


def operators(
    model: str,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **device_args: Any,
) -> None:
    """Print the model's operator list."""
    tflite_model, device = TFLiteModel(model), _get_device(**device_args)
    report(device, fmt="txt", space="bottom")

    operators = supported_operators(tflite_model, device)
    report([operators.ops, operators], fmt=output_format, output=output, space="top")


def performance(
    model: str,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    working_dir: Optional[str] = None,
    **device_args: Any,
) -> None:
    """Print model's performance stats."""
    tflite_model, device = TFLiteModel(model), _get_device(**device_args)
    report(device, fmt="txt", space="bottom")

    perf_metrics = collect_performance_metrics(tflite_model, device, working_dir)
    report(perf_metrics, fmt=output_format, output=output, space="top")


def model_optimization(
    model: str,
    out_path: Optional[str] = None,
    **optimizer_args: Any,
) -> str:
    """Apply specified optimization to the model and save resulting file."""
    keras_model = tf.keras.models.load_model(model)
    optimizer = _get_optimizer(keras_model, **optimizer_args)

    optimizer.apply_optimization()
    optimized_model = optimizer.get_model()
    optimized_model_path = save_keras_model(optimized_model, out_path)

    LOGGER.info(f"Model {model} saved to {optimized_model_path}")

    return optimized_model_path


def keras_to_tflite(
    model: str,
    out_path: str,
    quantized: bool,
) -> str:
    """Convert keras model to tflite format and save resulting file."""
    keras_model = tf.keras.models.load_model(model)
    tflite_model = convert_to_tflite(keras_model, quantized)
    tflite_model_path = save_tflite_model(tflite_model, out_path)

    LOGGER.info(f"Model {model} saved to {tflite_model_path}")

    return tflite_model_path


def _get_device(**kwargs: Any) -> EthosUConfiguration:
    device = kwargs.pop("device", None)
    if not device:
        raise Exception("Device is not provided")

    if device.lower() == "ethos-u55":
        return EthosU55(**kwargs)

    if device.lower() == "ethos-u65":
        return EthosU65(**kwargs)

    raise Exception(f"Unsupported device: {device}")


def _get_optimizer(model: tf.keras.Model, **kwargs: Any) -> Optimizer:
    optimization_target = kwargs.pop("optimization_target", None)
    if not optimization_target:
        raise Exception("Optimization target is not provided.")

    layers_to_optimize = kwargs.pop("layers_to_optimize", None)

    optimization_type = kwargs.pop("optimization_type", None)
    if not optimization_type:
        raise Exception("Optimization type is not provided")

    if optimization_type.lower() == "pruning":
        return Pruner(
            model, PruningConfiguration(optimization_target, layers_to_optimize)
        )
    elif optimization_type.lower() == "clustering":
        if optimization_target == int(optimization_target):
            return Clusterer(
                model,
                ClusteringConfiguration(int(optimization_target), layers_to_optimize),
            )
        raise Exception(
            f"""Optimization target should be a positive integer.
            Optimization target provided: {optimization_target}"""
        )

    raise Exception(f"Unsupported optimization type: {optimization_type}")
