# Copyright 2021, Arm Ltd.
"""CLI commands module.

This module contains functions which implement main app
functionality.

Before running them from scripts 'logging' module should
be configured. Function 'setup_logging' from module
'mli.cli.logging' could be used for that, e.g.

>>> from mlia.cli.logging import setup_logging
>>> setup_logging(verbose=True)
>>> import mlia.cli.commands as mlia
>>> mlia.all_tests("path/to/model", target="U55-256")
"""
import logging
from typing import List
from typing import Optional

from mlia.cli.advice import AdviceGroup
from mlia.cli.advice import AdvisorContext
from mlia.cli.advice import OptimizationResults
from mlia.cli.advice import produce_advice
from mlia.cli.options import parse_optimization_parameters
from mlia.core._typing import OutputFormat
from mlia.core._typing import PathOrFileLike
from mlia.core.context import ExecutionContext
from mlia.devices.ethosu.config import get_target
from mlia.devices.ethosu.operators import generate_supported_operators_report
from mlia.devices.ethosu.operators import supported_operators
from mlia.devices.ethosu.performance import collect_performance_metrics
from mlia.devices.ethosu.reporters import get_reporter
from mlia.nn.tensorflow.config import get_keras_model
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.optimizations.select import get_optimizer
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.use_cases import compare_metrics
from mlia.use_cases import optimize_and_compare

logger = logging.getLogger(__name__)


MODEL_ANALYSIS_MSG = """
=== Model Analysis =========================================================
"""

ADV_GENERATION_MSG = """
=== Advice Generation ======================================================
"""

REPORT_GENERATION_MSG = """
=== Report Generation ======================================================
"""


def all_tests(  # pylint: disable=too-many-arguments, too-many-locals
    ctx: ExecutionContext,
    target: str,
    model: str,
    optimization_type: str,
    optimization_target: str,
    output_format: OutputFormat = "plain_text",
    output: Optional[PathOrFileLike] = None,
) -> None:
    """Generate a full report on the input model.

    This command runs a series of tests in order to generate a
    comprehensive report/advice:

        - converts the input Keras model into TFLite format
        - checks the model for operator compatibility on the specified device
        - applies optimizations to the model and estimates the resulting performance
          on both the original and the optimized models
        - generates a final report on the steps above
        - provides advice on how to (possibly) improve the inference performance

    :param ctx: execution context
    :param target: target profile identifier. Will load appropriate parameters
            from the profile.json file based on this argument.
    :param model: path to the Keras model
    :param optimization_type: list of the optimization techniques separated
           by comma, e.g. 'pruning,clustering'
    :param optimization_target: list of the corresponding targets for
           the provided optimization techniques, e.g. '0.5,32'
    :param output_format: format of the report produced during the command
           execution
    :param output: path to the file where the report will be saved

    Example:
        Run command for the target profile U55-256 with two model optimizations
        and save report in json format locally in the file report.json

        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import all_tests
        >>> all_tests("model.h5", "pruning,clustering", "0.5,32",
                       output_format="json", output="report.json",
                       working_dir="mlia_output", target="U55-256")
    """
    keras_model = get_keras_model(model, ctx)
    device = get_target(target=target)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        logger.info(MODEL_ANALYSIS_MSG)

        converted_model_path = ctx.get_model_path("converted_model.tflite")
        tflite_model = keras_model.convert_to_tflite(
            converted_model_path, quantized=True
        )
        operators_info = supported_operators(tflite_model, device)

        logger.info("Evaluating performance ...\n")

        opt_params = parse_optimization_parameters(
            optimization_type, optimization_target
        )
        opt_settings = OptimizationSettings.create_from(opt_params)

        optimizer = get_optimizer(keras_model, opt_settings)
        original, optimized = optimize_and_compare(optimizer, device, ctx)

        reporter.submit([operators_info.ops, operators_info], space="top")

        reporter.submit(
            [original, optimized],
            columns_name="Metrics",
            title="Performance metrics",
            space=True,
            notes="IMPORTANT: The performance figures above refer to NPU only",
        )

        logger.info(ADV_GENERATION_MSG)

        adv_ctx = AdvisorContext(
            operators=operators_info,
            optimization_results=OptimizationResults(
                perf_metrics=compare_metrics(original, optimized),
                optimizations=opt_params,
            ),
            target=target,
            model=model,
        )
        advice = produce_advice(adv_ctx, AdviceGroup.COMMON)

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )
        if output is not None:
            logger.info(REPORT_GENERATION_MSG)
            logger.info("Report(s) and advice list saved to: %s", output)


def operators(  # pylint: disable=too-many-arguments
    ctx: ExecutionContext,
    target: str,
    model: Optional[str] = None,
    output_format: OutputFormat = "plain_text",
    output: Optional[PathOrFileLike] = None,
    supported_ops_report: bool = False,
) -> None:
    """Print the model's operator list.

    This command checks the operator compatibility of the input model with
    the specific target profile. Generates a report of the operator placement
    (NPU or CPU fallback) and advice on how to improve it (if necessary).

    :param ctx: execution context
    :param target: target profile identifier. Will load appropriate parameters
            from the profile.json file based on this argument.
    :param model: path to the model, which can be TFLite or Keras
    :param output_format: format of the report produced during the command
           execution
    :param output: path to the file where the report will be saved
    :param supported_ops_report: if True then generates supported operators
           report in current directory and exits

    Example:
        Run command for the target profile U55-256 and the provided TFLite model and
        print report on the standard output

        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import operators
        >>> operators("model.tflite", target="U55-256")
    """
    if supported_ops_report:
        generate_supported_operators_report()
        logger.info("Report saved into SUPPORTED_OPS.md")
        return

    if not model:
        raise Exception("Model is not provided")

    tflite_model = get_tflite_model(model, ctx)
    device = get_target(target=target)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        logger.info(MODEL_ANALYSIS_MSG)

        operators_info = supported_operators(tflite_model, device)
        reporter.submit([operators_info.ops, operators_info], space="top")

        logger.info(ADV_GENERATION_MSG)

        adv_ctx = AdvisorContext(operators=operators_info, target=target, model=model)
        advice = produce_advice(adv_ctx, AdviceGroup.OPERATORS_COMPATIBILITY)

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )
        if output is not None:
            logger.info(REPORT_GENERATION_MSG)
            logger.info("Report(s) and advice list saved to: %s", output)


def performance(
    ctx: ExecutionContext,
    target: str,
    model: str,
    output_format: OutputFormat = "plain_text",
    output: Optional[PathOrFileLike] = None,
) -> None:
    """Print the model's performance stats.

    This command estimates the inference performance of the input model
    on the specified target profile, and generates a report with advice on how
    to improve it.

    :param ctx: execution context
    :param target: target profile identifier. Will load appropriate parameters
            from the profile.json file based on this argument.
    :param model: path to the model, which can be TFLite or Keras
    :param output_format: format of the report produced during the command
           execution
    :param output: path to the file where the report will be saved

    Example:
        Run command for the target profile U55-256 and the provided TFLite model and
        print report on the standard output

        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import performance
        >>> performance("model.tflite", target="U55-256")
    """
    tflite_model = get_tflite_model(model, ctx)
    device = get_target(target=target)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        logger.info(MODEL_ANALYSIS_MSG)

        perf_metrics = collect_performance_metrics(tflite_model, device, ctx)
        reporter.submit(perf_metrics, space="top")

        logger.info(ADV_GENERATION_MSG)

        adv_ctx = AdvisorContext(perf_metrics=perf_metrics, target=target, model=model)
        advice = produce_advice(adv_ctx, AdviceGroup.PERFORMANCE)

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )

        if output is not None:
            logger.info(REPORT_GENERATION_MSG)
            logger.info("Report(s) and advice list saved to: %s", output)


def optimization(  # pylint: disable=too-many-arguments, too-many-locals
    ctx: ExecutionContext,
    target: str,
    model: str,
    optimization_type: str,
    optimization_target: str,
    layers_to_optimize: Optional[List[str]] = None,
    output_format: OutputFormat = "plain_text",
    output: Optional[PathOrFileLike] = None,
) -> None:
    """Show the performance improvements (if any) after applying the optimizations.

    This command applies the selected optimization techniques (up to the
    indicated targets) and generates a report with advice on how to improve
    the inference performance (if possible).

    :param ctx: execution context
    :param target: target profile identifier. Will load appropriate parameters
            from the profile.json file based on this argument.
    :param model: path to the TFLite model
    :param optimization_type: list of the optimization techniques separated
           by comma, e.g. 'pruning,clustering'
    :param optimization_target: list of the corresponding targets for
           the provided optimization techniques, e.g. '0.5,32'
    :param layers_to_optimize: list of the layers of the model which should be
           optimized, if None then all layers are used
    :param output_format: format of the report produced during the command
           execution
    :param output: path to the file where the report will be saved

    Example:
        Run command for the target profile U55-256 and the provided TFLite model and
        print report on the standard output

        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import optimization
        >>> optimization("model.tflite", target="U55-256")
    """
    keras_model = get_keras_model(model, ctx)
    device = get_target(target=target)

    with get_reporter(output_format, output) as reporter:
        reporter.submit(device)

        logger.info(MODEL_ANALYSIS_MSG)

        opt_params = parse_optimization_parameters(
            optimization_type, optimization_target
        )
        opt_settings = OptimizationSettings.create_from(opt_params, layers_to_optimize)

        optimizer = get_optimizer(keras_model, opt_settings)
        original, optimized = optimize_and_compare(optimizer, device, ctx)

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

        logger.info(ADV_GENERATION_MSG)

        optimization_results = OptimizationResults(
            perf_metrics=compare_metrics(original, optimized),
            optimizations=opt_params,
        )

        adv_ctx = AdvisorContext(
            optimization_results=optimization_results,
            target=target,
            model=model,
        )
        advice = produce_advice(adv_ctx, AdviceGroup.OPTIMIZATION)

        reporter.submit(
            advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )

        if output is not None:
            logger.info(REPORT_GENERATION_MSG)
            logger.info("Report(s) and advice list saved to: %s", output)
