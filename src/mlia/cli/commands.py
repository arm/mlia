# Copyright (C) 2021-2022, Arm Ltd.
"""CLI commands module.

This module contains functions which implement main app
functionality.

Before running them from scripts 'logging' module should
be configured. Function 'setup_logging' from module
'mli.cli.logging' could be used for that, e.g.

>>> from mlia.api import ExecutionContext
>>> from mlia.cli.logging import setup_logging
>>> setup_logging(verbose=True)
>>> import mlia.cli.commands as mlia
>>> mlia.all_tests(ExecutionContext(working_dir="mlia_output"), "U55-256",
                   "path/to/model")
"""
import logging
from typing import List
from typing import Optional

from mlia.api import ExecutionContext
from mlia.api import get_advice
from mlia.api import PathOrFileLike
from mlia.cli.options import parse_optimization_parameters
from mlia.devices.ethosu.operators import generate_supported_operators_report

logger = logging.getLogger(__name__)


def all_tests(
    ctx: ExecutionContext,
    target: str,
    model: str,
    optimization_type: str = "pruning,clustering",
    optimization_target: str = "0.5,32",
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
    :param output: path to the file where the report will be saved

    Example:
        Run command for the target profile U55-256 with two model optimizations
        and save report in json format locally in the file report.json

        >>> from mlia.api import ExecutionContext
        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import all_tests
        >>> all_tests(ExecutionContext(working_dir="mlia_output"), "U55-256",
                      "model.h5", "pruning,clustering", "0.5,32",
                       output="report.json")
    """
    opt_params = parse_optimization_parameters(
        optimization_type,
        optimization_target,
    )

    get_advice(
        target,
        model,
        "all",
        optimization_targets=opt_params,
        output=output,
        context=ctx,
    )


def operators(
    ctx: ExecutionContext,
    target: str,
    model: Optional[str] = None,
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
    :param output: path to the file where the report will be saved
    :param supported_ops_report: if True then generates supported operators
           report in current directory and exits

    Example:
        Run command for the target profile U55-256 and the provided TFLite model and
        print report on the standard output

        >>> from mlia.api import ExecutionContext
        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import operators
        >>> operators(ExecutionContext(working_dir="mlia_output"), "U55-256",
                      "model.tflite")
    """
    if supported_ops_report:
        generate_supported_operators_report()
        logger.info("Report saved into SUPPORTED_OPS.md")
        return

    if not model:
        raise Exception("Model is not provided")

    get_advice(
        target,
        model,
        "operators",
        output=output,
        context=ctx,
    )


def performance(
    ctx: ExecutionContext,
    target: str,
    model: str,
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
    :param output: path to the file where the report will be saved

    Example:
        Run command for the target profile U55-256 and the provided TFLite model and
        print report on the standard output

        >>> from mlia.api import ExecutionContext
        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import performance
        >>> performance(ExecutionContext(working_dir="mlia_output"), "U55-256",
                        "model.tflite")
    """
    get_advice(
        target,
        model,
        "performance",
        output=output,
        context=ctx,
    )


def optimization(
    ctx: ExecutionContext,
    target: str,
    model: str,
    optimization_type: str,
    optimization_target: str,
    layers_to_optimize: Optional[List[str]] = None,
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
    :param output: path to the file where the report will be saved

    Example:
        Run command for the target profile U55-256 and the provided TFLite model and
        print report on the standard output

        >>> from mlia.cli.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import optimization
        >>> optimization(ExecutionContext(working_dir="mlia_output"), target="U55-256",
                         "model.tflite", "pruning", "0.5")
    """
    opt_params = parse_optimization_parameters(
        optimization_type,
        optimization_target,
        layers_to_optimize=layers_to_optimize,
    )

    get_advice(
        target,
        model,
        "optimization",
        optimization_targets=opt_params,
        output=output,
        context=ctx,
    )
