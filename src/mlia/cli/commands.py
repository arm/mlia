# Copyright 2021, Arm Ltd.
"""Cli commands module."""
import sys
from typing import Any

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
from mlia.utils.general import save_keras_model


def operators(
    model: str,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **device_args: Any,
) -> None:
    """Print the model's operator list."""
    tflite_model, device = TFLiteModel(model), _get_device(**device_args)
    ops = supported_operators(tflite_model, device)

    report([device, ops], fmt=output_format, output=output)


def performance(
    model: str,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **device_args: Any,
) -> None:
    """Print model's performance stats."""
    tflite_model, device = TFLiteModel(model), _get_device(**device_args)
    perf_metrics = collect_performance_metrics(tflite_model, device)

    report([device, perf_metrics], fmt=output_format, output=output)


def model_optimization(model: str, **optimizer_args: Any) -> None:
    """Apply specified optimization to the model and save resulting file."""
    keras_model = tf.keras.models.load_model(model)
    optimizer = _get_optimizer(keras_model, **optimizer_args)

    optimizer.apply_optimization()
    optimized_model = optimizer.get_model()

    saved_model_path = save_keras_model(optimized_model)

    print(f"Model {model} saved to {saved_model_path}")


def _get_device(**kwargs: Any) -> EthosUConfiguration:
    device = kwargs.pop("device", None)
    if not device:
        raise Exception("Device is not provided")

    if device.lower() == "ethos-u55":
        return EthosU55(**kwargs)
    elif device.lower() == "ethos-u65":
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
