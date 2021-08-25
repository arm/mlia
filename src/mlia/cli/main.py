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
from mlia.cli.commands import keras_to_tflite
from mlia.cli.commands import operators
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.cli.options import add_custom_supported_operators_options
from mlia.cli.options import add_debug_options
from mlia.cli.options import add_device_options
from mlia.cli.options import add_keras_model_options
from mlia.cli.options import add_optimization_options
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


def setup_logging(logs_dir: str, verbose: bool = False) -> None:
    """Set up loggers."""
    logs_dir_path = Path(logs_dir)
    if not logs_dir_path.exists():
        logs_dir_path.mkdir()

    default_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    now = datetime.datetime.now()
    timestamp = f"{now:%Y%m%d_%H%M%S}"

    for logger_name in [
        "mlia.tools.aiet",
        "mlia.tools.vela",
        "tensorflow",
        "py.warnings",
    ]:
        module_name = logger_name.split(".")[-1]

        handler = logging.FileHandler(
            logs_dir_path / f"{timestamp}_{module_name}.log", delay=True
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(default_formatter)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        if verbose:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
            logger.addHandler(stdout_handler)

    for logger_name in ["mlia.cli", "mlia.performance", "mlia.operators"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))


def init_commands(parser: argparse.ArgumentParser) -> None:
    """Init cli subcommands."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    commands = [
        (
            operators,
            ["ops"],
            [
                add_device_options,
                add_output_options,
                add_custom_supported_operators_options,
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
            keras_to_tflite,
            ["k2l"],
            [add_keras_model_options, add_quantize_option, add_out_path],
        ),
        (
            optimization,
            ["opt"],
            [
                add_keras_model_options,
                add_optimization_options,
                add_device_options,
                add_debug_options,
            ],
        ),
    ]

    for command in commands:
        func, aliases, opt_groups = command
        command_parser = subparsers.add_parser(
            func.__name__, aliases=aliases, help=func.__doc__
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
        description=INFO_MESSAGE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s { __version__}"
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
