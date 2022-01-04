# Copyright 2021, Arm Ltd.
"""CLI main entry point."""
import argparse
import datetime
import logging
import sys
from inspect import signature
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from mlia import __version__
from mlia.cli.commands import all_tests
from mlia.cli.commands import operators
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.cli.common import CommandInfo
from mlia.cli.common import ExecutionContext
from mlia.cli.logging import setup_logging
from mlia.cli.options import add_custom_supported_operators_options
from mlia.cli.options import add_debug_options
from mlia.cli.options import add_keras_model_options
from mlia.cli.options import add_multi_optimization_options
from mlia.cli.options import add_optional_tflite_model_options
from mlia.cli.options import add_output_options
from mlia.cli.options import add_target_options
from mlia.cli.options import add_tflite_model_options


logger = logging.getLogger(__name__)

INFO_MESSAGE = f"""
ML Inference Advisor {__version__}

Help the design and optimization of neural network models for efficient inference on a target CPU, GPU and NPU

Supported targets:

 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>

ARM {datetime.datetime.now():%Y} Copyright Reserved
"""


def get_commands() -> List[CommandInfo]:
    """Return commands configuration."""
    return [
        CommandInfo(
            all_tests,
            ["all"],
            [
                add_target_options,
                add_keras_model_options,
                add_multi_optimization_options,
                add_output_options,
                add_debug_options,
            ],
            True,
        ),
        CommandInfo(
            operators,
            ["ops"],
            [
                add_target_options,
                add_optional_tflite_model_options,
                add_output_options,
                add_custom_supported_operators_options,
                add_debug_options,
            ],
        ),
        CommandInfo(
            performance,
            ["perf"],
            [
                add_target_options,
                add_tflite_model_options,
                add_output_options,
                add_debug_options,
            ],
        ),
        CommandInfo(
            optimization,
            ["opt"],
            [
                add_target_options,
                add_keras_model_options,
                add_multi_optimization_options,
                add_output_options,
                add_debug_options,
            ],
        ),
    ]


def get_default_command() -> Optional[str]:
    """Get name of the default command."""
    commands = get_commands()

    marked_as_default = [cmd.command_name for cmd in commands if cmd.is_default]
    assert len(marked_as_default) <= 1, "Only one command could be marked as default"

    return next(iter(marked_as_default), None)


def get_possible_command_names() -> List[str]:
    """Get all possible command names including aliases."""
    return [
        name_or_alias
        for cmd in get_commands()
        for name_or_alias in cmd.command_name_and_aliases
    ]


def init_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Init cli subcommands."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    for command in get_commands():
        command_parser = subparsers.add_parser(
            command.command_name,
            aliases=command.aliases,
            help=command.command_help,
        )
        command_parser.set_defaults(func=command.func)
        for opt_group in command.opt_groups:
            opt_group(command_parser)

    return parser


def setup_context(
    args: argparse.Namespace, context_var_name: str = "ctx"
) -> Tuple[ExecutionContext, Dict]:
    """Set up context and resolve function parameters."""
    ctx = ExecutionContext(
        working_dir=args.working_dir,
        verbose="verbose" in args and args.verbose,
    )

    # these parameters should not be passed into command function
    skipped_params = ["func", "command", "working_dir", "verbose"]

    # pass these parameters only if command expects them
    expected_params = [context_var_name]
    func_params = signature(args.func).parameters

    params = {context_var_name: ctx, **vars(args)}

    func_args = {
        param_name: param_value
        for param_name, param_value in params.items()
        if param_name not in skipped_params
        and (param_name not in expected_params or param_name in func_params)
    }

    return (ctx, func_args)


def run_command(args: argparse.Namespace) -> int:
    """Run command."""
    ctx, func_args = setup_context(args)
    setup_logging(ctx.logs_path, ctx.verbose)

    logger.debug(
        "*** This is the beginning of the command '%s' execution ***", args.command
    )

    try:
        logger.info(INFO_MESSAGE)

        args.func(**func_args)
        return 0
    except KeyboardInterrupt:
        logger.error("Execution has been interrupted")
    except Exception as err:  # pylint: disable=broad-except
        logger.error(
            "Execution failed with the error: '%s'",
            err,
            exc_info=err if ctx.verbose else None,
        )

        err_advice_message = (
            f"Please check the log files in the {ctx.logs_path} "
            "directory for more details"
        )
        if not ctx.verbose:
            err_advice_message += ", or enable verbose mode"

        logger.error(err_advice_message)

    return 1


def init_common_parser() -> argparse.ArgumentParser:
    """Init common parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--working-dir",
        default=f"{Path.cwd() / 'mlia_output'}",
        help="Path to the directory where MLIA will store logs, "
        "models, etc. (default: %(default)s)",
    )

    return parser


def init_subcommand_parser(parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Init subcommand parser."""
    parser = argparse.ArgumentParser(
        description=INFO_MESSAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent],
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
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit",
    )

    return parser


def add_default_command_if_needed(args: List[str]) -> None:
    """Add default command to the list of the arguments if needed."""
    default_command = get_default_command()

    if default_command and len(args) > 0:
        commands = get_possible_command_names()
        help_or_version = ["-h", "--help", "-v", "--version"]

        command_is_missing = args[0] not in [*commands, *help_or_version]
        if command_is_missing:
            args.insert(0, default_command)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the application."""
    common_parser = init_common_parser()
    subcommand_parser = init_subcommand_parser(common_parser)
    init_commands(subcommand_parser)

    common_args, subcommand_args = common_parser.parse_known_args(argv)
    add_default_command_if_needed(subcommand_args)

    args = subcommand_parser.parse_args(subcommand_args, common_args)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
