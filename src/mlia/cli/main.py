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
from mlia.cli.commands import estimate_optimized_performance
from mlia.cli.commands import keras_to_tflite
from mlia.cli.commands import model_optimization
from mlia.cli.commands import operators
from mlia.cli.commands import performance

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

    for logger_name in ["mlia.tools.aiet", "mlia.tools.vela"]:
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


def add_device_options(parser: argparse.ArgumentParser) -> None:
    """Add device specific options."""
    device_group = parser.add_argument_group("device options")
    device_group.add_argument(
        "--device",
        choices=("ethos-u55", "ethos-u65"),
        default="ethos-u55",
        help="Device type (default: %(default)s)",
    )
    device_group.add_argument(
        "--mac",
        choices=[32, 64, 128, 256, 512],
        type=int,
        default=256,
        help="MAC value (default: %(default)s)",
    )
    device_group.add_argument(
        "--config",
        type=str,
        action="append",
        dest="config_files",
        help="Vela configuration file(s) in Python ConfigParser .ini file format",
    )
    device_group.add_argument(
        "--system-config",
        default="internal-default",
        help="System configuration (default: %(default)s)",
    )
    device_group.add_argument(
        "--memory-mode",
        default="internal-default",
        help="Memory mode (default: %(default)s)",
    )
    device_group.add_argument(
        "--max-block-dependency",
        type=int,
        default=3,
        help="Max block dependency (default: %(default)s)",
    )
    device_group.add_argument("--arena-cache-size", type=int, help="Arena cache size")
    device_group.add_argument(
        "--tensor-allocator",
        choices=("LinearAlloc", "Greedy", "HillClimb"),
        default="HillClimb",
        help="Tensor allocator algorithm",
    )
    device_group.add_argument(
        "--cpu-tensor-alignment",
        type=int,
        default=16,
        help="CPU tensor alignment (default: %(default)s)",
    )
    device_group.add_argument(
        "--optimization-strategy",
        choices=("Performance", "Size"),
        default="Performance",
        help="Optimization strategy (default: %(default)s)",
    )


def add_optimization_options(parser: argparse.ArgumentParser) -> None:
    """Add optimization specific options."""
    optimization_group = parser.add_argument_group("optimization options")

    optimization_group.add_argument(
        "--optimization-type",
        required=True,
        choices=("pruning", "clustering"),
        help="Optimization type [required]",
    )
    optimization_group.add_argument(
        "--optimization-target",
        required=True,
        type=float,
        help="""Target for optimization
            (for pruning this is sparsity between (0,1),
            for clustering this is the number of clusters (positive integer))
            [required]""",
    )
    optimization_group.add_argument(
        "--layers-to-optimize",
        nargs="+",
        type=str,
        help="""Name of the layers to optimize (separated by space)
            example: conv1 conv2 conv3
            [default: every layer]""",
    )


def add_tflite_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("TFLite model options")
    model_group.add_argument("model", help="TFLite model")


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output specific options."""
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output-format",
        choices=["txt", "json", "csv"],
        default="txt",
        help="Output format (default: %(default)s)",
    )
    output_group.add_argument(
        "--output",
        default=sys.stdout,
        help=(
            "Name of the file where report will be saved. If no file "
            "name is specified, the report will be displayed on the standard output"
        ),
    )


def add_debug_options(parser: argparse.ArgumentParser) -> None:
    """Add debug options."""
    debug_group = parser.add_argument_group("debug options")
    debug_group.add_argument(
        "--verbose", default=False, action="store_true", help="Produce verbose output"
    )


def add_keras_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("Keras model options")
    model_group.add_argument("model", help="Keras model")


def add_quantize_option(parser: argparse.ArgumentParser) -> None:
    """Add quantization if needed."""
    quantization_group = parser.add_argument_group(
        "quantization_opts", "Quantization options"
    )
    quantization_group.add_argument(
        "--quantized",
        default=False,
        action="store_true",
        help="""Quantizes model to int8 if provided.
        Leaves it as fp32 if otherwise.""",
    )


def add_out_path(parser: argparse.ArgumentParser) -> None:
    """Add option for output path instead of temporary directory."""
    out_path_group = parser.add_argument_group("out_path", "Output path")
    out_path_group.add_argument("--out_path", default=None)


def add_custom_supported_operators_options(parser: argparse.ArgumentParser) -> None:
    """Add custom options for the command 'operators'."""
    parser.add_argument(
        "--supported-ops-report",
        action="store_true",
        default=False,
        help=(
            "Generate the SUPPORTED_OPS.md file in the "
            "current working directory and exit"
        ),
    )

    model_group = parser.add_argument_group("TFLite model options")
    # make model parameter optional
    model_group.add_argument("model", nargs="?", help="TFLite model")


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
            model_optimization,
            ["mopt"],
            [add_keras_model_options, add_optimization_options, add_out_path],
        ),
        (
            keras_to_tflite,
            ["k2l"],
            [add_keras_model_options, add_out_path, add_quantize_option],
        ),
        (
            estimate_optimized_performance,
            ["eop"],
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
        default="mlia_output",
        help="Path to the directory where MLIA will store logs, models, etc",
    )
    init_commands(parser)

    args = parser.parse_args(argv)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
