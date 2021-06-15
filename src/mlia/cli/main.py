"""Cli main entry point."""
import argparse
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import get_type_hints
from typing import List
from typing import Optional
from typing import Tuple

from mlia import __version__
from mlia.cli.commands import operators
from mlia.cli.commands import ParamAnnotation
from mlia.cli.commands import performance


def params(command: Callable) -> Dict[str, Tuple[type, ParamAnnotation]]:
    """Get list of the parameters for the command."""
    return {
        name: (hint.__origin__, hint.__metadata__[0])
        for name, hint in get_type_hints(command).items()
        if hasattr(hint, "__metadata__")  # take only annotated params
    }


def init_commands(parser: argparse.ArgumentParser) -> None:
    """Init cli subcommands."""
    commands = [operators, performance]

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    for command in commands:
        name = command.__name__

        command_parser = subparsers.add_parser(name, help=command.__doc__)
        for name, (param_type, annotation) in params(command).items():
            arg_name = name if annotation.positional else "--{}".format(name)

            arg_params: Dict[str, Any] = {"help": annotation.description}
            if not annotation.positional and annotation.required:
                arg_params["required"] = annotation.required

            if param_type in [int, str]:
                arg_params["type"] = param_type

            command_parser.add_argument(arg_name, **arg_params)
        command_parser.set_defaults(func=command)


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
        "--version", action="version", version="%(prog)s " + __version__
    )

    init_commands(parser)
    args = parser.parse_args(argv)
    run_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
