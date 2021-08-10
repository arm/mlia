# Copyright 2021, Arm Ltd.
"""CLI commands module."""
import logging
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.cli.advice import AdviceGroup
from mlia.cli.advice import AdvisorContext
from mlia.cli.advice import show_advice
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import KerasModel
from mlia.config import TFLiteModel
from mlia.operators import generate_supported_operators_report
from mlia.operators import supported_operators
from mlia.optimizations.clustering import Clusterer
from mlia.optimizations.clustering import ClusteringConfiguration
from mlia.optimizations.common import Optimizer
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.pruning import PruningConfiguration
from mlia.performance import collect_performance_metrics
from mlia.reporters import report
from mlia.use_cases import optimize_and_compare
from mlia.use_cases import optimize_model
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import save_tflite_model

LOGGER = logging.getLogger("mlia.cli")


def operators(
    model: Optional[str] = None,
    output_format: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    supported_ops_report: bool = False,
    **device_args: Any,
) -> None:
    """Print the model's operator list."""
    if supported_ops_report:
        generate_supported_operators_report()
        LOGGER.info("Report saved into SUPPORTED_OPS.md")
        return

    if not model:
        raise Exception("Model is not provided")

    tflite_model, device = TFLiteModel(model), _get_device(**device_args)
    report(device, fmt="txt", space="bottom")

    operators = supported_operators(tflite_model, device)
    report([operators.ops, operators], fmt=output_format, output=output, space="top")

    show_advice(
        AdvisorContext(operators=operators, device_args=device_args, model=model),
        advice_group=AdviceGroup.OPERATORS_COMPATIBILITY,
    )


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
    optimization_type: str,
    optimization_target: Union[int, float],
    layers_to_optimize: Optional[List[str]] = None,
    out_path: Optional[str] = None,
) -> None:
    """Apply specified optimization to the model and save resulting file."""
    optimizer = _get_optimizer(
        model, optimization_type, optimization_target, layers_to_optimize
    )

    optimized_model = optimize_model(optimizer)
    optimized_model_path = save_tflite_model(optimized_model, out_path)

    print(f"Model {model} saved to {optimized_model_path}")


def keras_to_tflite(
    model: str, quantized: bool, out_path: Optional[str] = None
) -> None:
    """Convert keras model to tflite format and save resulting file."""
    tflite_model = convert_to_tflite(KerasModel(model).get_keras_model(), quantized)
    tflite_model_path = save_tflite_model(tflite_model, out_path)

    LOGGER.info(f"Model {model} saved to {tflite_model_path}")


def estimate_optimized_performance(
    model: str,
    optimization_type: str,
    optimization_target: Union[int, float],
    layers_to_optimize: Optional[List[str]] = None,
    **device_args: Any,
) -> None:
    """Show performance improvements after applying optimizations."""
    optimizer = _get_optimizer(
        model, optimization_type, optimization_target, layers_to_optimize
    )
    device = _get_device(**device_args)

    results = optimize_and_compare(optimizer, device)

    report(results, columns_name="Metrics")


def _get_device(**kwargs: Any) -> EthosUConfiguration:
    device = kwargs.pop("device", None)
    if not device:
        raise Exception("Device is not provided")

    if device.lower() == "ethos-u55":
        return EthosU55(**kwargs)

    if device.lower() == "ethos-u65":
        return EthosU65(**kwargs)

    raise Exception(f"Unsupported device: {device}")


def _get_optimizer(
    model: str,
    optimization_type: str,
    optimization_target: Union[int, float],
    layers_to_optimize: Optional[List[str]] = None,
) -> Optimizer:
    if not optimization_target:
        raise Exception("Optimization target is not provided.")

    if not optimization_type:
        raise Exception("Optimization type is not provided")

    if optimization_type.lower() == "pruning":
        return Pruner(
            KerasModel(model).get_keras_model(),
            PruningConfiguration(optimization_target, layers_to_optimize),
        )
    elif optimization_type.lower() == "clustering":
        # make sure an integer is given as clustering target
        if optimization_target == int(optimization_target):
            return Clusterer(
                KerasModel(model).get_keras_model(),
                ClusteringConfiguration(int(optimization_target), layers_to_optimize),
            )
        raise Exception(
            f"""Optimization target should be a positive integer.
            Optimization target provided: {optimization_target}"""
        )

    raise Exception(f"Unsupported optimization type: {optimization_type}")
