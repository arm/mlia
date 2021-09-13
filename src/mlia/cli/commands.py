# Copyright 2021, Arm Ltd.
"""CLI commands module."""
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.cli.advice import AdviceGroup
from mlia.cli.advice import AdvisorContext
from mlia.cli.advice import OptimizationResults
from mlia.cli.advice import produce_advice
from mlia.cli.options import parse_optimizer_params
from mlia.config import get_device
from mlia.config import KerasModel
from mlia.config import TFLiteModel
from mlia.operators import generate_supported_operators_report
from mlia.operators import supported_operators
from mlia.optimizations.select import get_optimizer
from mlia.optimizations.select import OptimizationSettings
from mlia.performance import collect_performance_metrics
from mlia.reporters import get_reporter
from mlia.use_cases import compare_metrics
from mlia.use_cases import optimize_and_compare
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import save_tflite_model

LOGGER = logging.getLogger("mlia.cli")


MODEL_ANALYSIS_MSG = """
=== Model Analysis =========================================================
"""

ADV_GENERATION_MSG = """
=== Advice Generation ======================================================
"""


def all_tests(
    model: str,
    optimization_type: str,
    optimization_target: str,
    output_format: OutputFormat = "plain_text",
    output: PathOrFileLike = sys.stdout,
    working_dir: Optional[str] = None,
    **device_args: Any,
) -> None:
    """Generate full report."""
    models_path = Path(working_dir) if working_dir else Path.cwd()
    keras_model, device = KerasModel(model), get_device(**device_args)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        LOGGER.info(MODEL_ANALYSIS_MSG)

        converted_model = convert_to_tflite(keras_model.get_keras_model(), True)
        converted_model_path = models_path / "converted_model.tflite"
        save_tflite_model(converted_model, converted_model_path)

        tflite_model = TFLiteModel(converted_model_path)
        operators = supported_operators(tflite_model, device)

        LOGGER.info("Evaluating performance ...\n")

        optimizer_params = parse_optimizer_params(
            optimization_type, optimization_target
        )

        optimizer_settings = [
            OptimizationSettings(
                optimization_type=opt_type,
                optimization_target=opt_target,
                layers_to_optimize=None,
            )
            for opt_type, opt_target in optimizer_params
        ]

        keras_model = KerasModel(model).get_keras_model()
        optimizer = get_optimizer(keras_model, optimizer_settings)
        original, optimized = optimize_and_compare(optimizer, device, working_dir)

        reporter.submit([operators.ops, operators], space="top")

        reporter.submit(
            [original, optimized],
            columns_name="Metrics",
            title="Performance metrics",
            space=True,
            notes="IMPORTANT: The performance figures above refer to NPU only",
        )

        LOGGER.info(ADV_GENERATION_MSG)

        advice = produce_advice(
            AdvisorContext(
                operators=operators,
                optimization_results=OptimizationResults(
                    perf_metrics=compare_metrics(original, optimized),
                    optimizations=optimizer_params,
                ),
                device_args=device_args,
                model=model,
            ),
            AdviceGroup.COMMON,
        )

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )


def operators(
    model: Optional[str] = None,
    output_format: OutputFormat = "plain_text",
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

    tflite_model, device = TFLiteModel(model), get_device(**device_args)
    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        LOGGER.info(MODEL_ANALYSIS_MSG)

        operators = supported_operators(tflite_model, device)
        reporter.submit([operators.ops, operators], space="top")

        LOGGER.info(ADV_GENERATION_MSG)

        advice = produce_advice(
            AdvisorContext(operators=operators, device_args=device_args, model=model),
            advice_group=AdviceGroup.OPERATORS_COMPATIBILITY,
        )

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )


def performance(
    model: str,
    output_format: OutputFormat = "plain_text",
    output: PathOrFileLike = sys.stdout,
    working_dir: Optional[str] = None,
    **device_args: Any,
) -> None:
    """Print model's performance stats."""
    if os.path.splitext(model)[1] != ".tflite":
        raise ValueError(
            "The input model format for the performance estimation must be tflite!"
        )
    tflite_model, device = TFLiteModel(model), get_device(**device_args)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        LOGGER.info(MODEL_ANALYSIS_MSG)

        perf_metrics = collect_performance_metrics(tflite_model, device, working_dir)
        reporter.submit(perf_metrics, space="top")

        LOGGER.info(ADV_GENERATION_MSG)

        advice = produce_advice(
            AdvisorContext(
                perf_metrics=perf_metrics, device_args=device_args, model=model
            ),
            advice_group=AdviceGroup.PERFORMANCE,
        )

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )


def keras_to_tflite(
    model: str, quantized: bool, out_path: Optional[str] = None
) -> None:
    """Convert keras model to tflite format and save resulting file."""
    out_path_final = Path(out_path) if out_path else Path().cwd()

    tflite_model = convert_to_tflite(KerasModel(model).get_keras_model(), quantized)
    save_tflite_model(tflite_model, out_path_final / "converted_model.tflite")

    LOGGER.info(f"Model {model} saved to {out_path_final}")


def optimization(
    model: str,
    optimization_type: str,
    optimization_target: Union[int, float],
    layers_to_optimize: Optional[List[str]] = None,
    output_format: OutputFormat = "plain_text",
    output: PathOrFileLike = sys.stdout,
    working_dir: Optional[str] = None,
    **device_args: Any,
) -> None:
    """Show performance improvements after applying optimizations."""
    keras_model = KerasModel(model).get_keras_model()
    optimizer = get_optimizer(
        keras_model,
        OptimizationSettings(
            optimization_type, optimization_target, layers_to_optimize
        ),
    )
    device = get_device(**device_args)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        LOGGER.info(MODEL_ANALYSIS_MSG)

        original, optimized = optimize_and_compare(optimizer, device, working_dir)

        reporter.submit(
            [original, optimized],
            columns_name="Metrics",
            title="Performance metrics",
            space=True,
            notes=(
                "IMPORTANT: The applied tooling techniques have an impact "
                "on accuracy. Additional hyperparameter tuning may be required "
                "after any optimization."
            ),
        )

        LOGGER.info(ADV_GENERATION_MSG)

        advice = produce_advice(
            AdvisorContext(
                optimization_results=OptimizationResults(
                    perf_metrics=compare_metrics(original, optimized),
                    optimizations=[(optimization_type, optimization_target)],
                ),
                device_args=device_args,
                model=model,
            ),
            advice_group=AdviceGroup.OPTIMIZATION,
        )

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )
