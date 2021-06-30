"""Cli main entry point."""
import argparse
import sys
from typing import List
from typing import Optional

from mlia import __version__
from mlia.cli.commands import operators
from mlia.cli.commands import performance


def add_device_options(parser: argparse.ArgumentParser) -> None:
    """Add device specific options."""
    device_group = parser.add_argument_group("device_opts", "Device options")
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
        "--tensor_allocator",
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


def add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("model_opts", "Model options")
    model_group.add_argument("model", help="TFLite model")


def init_commands(parser: argparse.ArgumentParser) -> None:
    """Init cli subcommands."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    commands = [
        (operators, ["ops"], [add_device_options, add_model_options]),
        (performance, ["perf"], [add_device_options, add_model_options]),
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
