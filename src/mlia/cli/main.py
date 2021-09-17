# Copyright 2021, Arm Ltd.
"""CLI main entry point."""
import argparse
import datetime
import logging
import sys
from inspect import signature
from pathlib import Path
from typing import List
from typing import Optional

from mlia import __version__
from mlia.cli.commands import all_tests
from mlia.cli.commands import keras_to_tflite
from mlia.cli.commands import operators
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.cli.logging import setup_logging
from mlia.cli.options import add_custom_supported_operators_options
from mlia.cli.options import add_debug_options
from mlia.cli.options import add_device_options
from mlia.cli.options import add_keras_model_options
from mlia.cli.options import add_multi_optimization_options
from mlia.cli.options import add_optimization_options
from mlia.cli.options import add_optional_tflite_model_options
from mlia.cli.options import add_out_path
from mlia.cli.options import add_output_options
from mlia.cli.options import add_quantize_option
from mlia.cli.options import add_tflite_model_options


LOGGER = logging.getLogger("mlia.cli")

INFO_MESSAGE = f"""
ML Inference Advisor {__version__}

Help the design and optimization of neural network models for efficient inference on a target CPU, GPU and NPU

Supported targets:

 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>

ARM {datetime.datetime.now():%Y} Copyright Reserved
"""


def init_commands(parser: argparse.ArgumentParser) -> None:
    """Init cli subcommands."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    commands = [
        (
            all_tests,
            ["all"],
            [
                add_device_options,
                add_keras_model_options,
                add_multi_optimization_options,
                add_output_options,
                add_debug_options,
            ],
        ),
        (
            operators,
            ["ops"],
            [
                add_device_options,
                add_optional_tflite_model_options,
                add_output_options,
                add_custom_supported_operators_options,
                add_debug_options,
            ],
        ),
        (
            performance,
            ["perf"],
            [
                add_device_options,
                add_tflite_model_options,
                add_output_options,
                add_debug_options,
            ],
        ),
        (
            optimization,
            ["opt"],
            [
                add_device_options,
                add_keras_model_options,
                add_optimization_options,
                add_output_options,
                add_debug_options,
            ],
        ),
        (
            keras_to_tflite,
            ["k2l"],
            [add_keras_model_options, add_quantize_option, add_out_path],
        ),
    ]

    for command in commands:
        func, aliases, opt_groups = command
        assert func.__doc__, "Command function does not have a docstring"
        first_doc_line = func.__doc__.splitlines()[0]

        command_parser = subparsers.add_parser(
            func.__name__, aliases=aliases, help=first_doc_line
        )
        command_parser.set_defaults(func=func)
        for opt_group in opt_groups:
            opt_group(command_parser)


def run_command(args: argparse.Namespace) -> int:
    """Run command."""
    result = 1
    try:
        verbose = "verbose" in args and args.verbose

        setup_logging(args.working_dir, verbose)
        LOGGER.info(INFO_MESSAGE)

        # these parameters should not be passed into command function
        skipped_params = ["func", "command", "verbose"]

        # pass these parameters only if command expects them
        expected_params = ["working_dir"]
        func_params = signature(args.func).parameters

        kwargs = {
            param_name: param_value
            for param_name, param_value in vars(args).items()
            if param_name not in skipped_params
            and (param_name not in expected_params or param_name in func_params)
        }
        args.func(**kwargs)
    except KeyboardInterrupt:
        LOGGER.error("Execution has been interrupted")
    except Exception as e:
        LOGGER.error(
            f"Execution failed with error: {e}. Please check the log files in the "
            f"{args.working_dir} directory for more details, or enable verbose mode",
            exc_info=e if verbose else None,
        )
    else:
        result = 0

    return result


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the application."""
    parser = argparse.ArgumentParser(
        description=INFO_MESSAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s { __version__}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--working-dir",
        default=f"{Path.cwd() / 'mlia_output'}",
        help="Path to the directory where MLIA will store logs, "
        "models, etc. (default: %(default)s)",
    )
    init_commands(parser)

    args = parser.parse_args(argv)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
