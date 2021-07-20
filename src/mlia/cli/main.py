# Copyright 2021, Arm Ltd.
"""CLI main entry point."""
import argparse
import sys
from typing import List
from typing import Optional

from mlia import __version__
from mlia.cli.commands import keras_to_tflite
from mlia.cli.commands import model_optimization
from mlia.cli.commands import operators
from mlia.cli.commands import performance


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


def init_commands(parser: argparse.ArgumentParser) -> None:
    """Init cli subcommands."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    commands = [
        (
            operators,
            ["ops"],
            [add_device_options, add_tflite_model_options, add_output_options],
        ),
        (
            performance,
            ["perf"],
            [add_device_options, add_tflite_model_options, add_output_options],
        ),
        (
            model_optimization,
            ["mopt"],
            [add_keras_model_options, add_optimization_options],
        ),
        (
            keras_to_tflite,
            ["k2l"],
            [add_keras_model_options, add_out_path, add_quantize_option],
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


def run_command(args: argparse.Namespace) -> None:
    """Run command."""
    kwargs = {
        param_name: param_value
        for param_name, param_value in args.__dict__.items()
        if param_name not in ["func", "command"]
    }
    args.func(**kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the application."""
    parser = argparse.ArgumentParser(
        description="ML Inference advisor command line tool",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s { __version__}"
    )
    init_commands(parser)

    args = parser.parse_args(argv)
    run_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
