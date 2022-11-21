# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI main entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from inspect import signature
from pathlib import Path

from mlia import __version__
from mlia.cli.commands import all_tests
from mlia.cli.commands import backend_install
from mlia.cli.commands import backend_list
from mlia.cli.commands import backend_uninstall
from mlia.cli.commands import operators
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.cli.common import CommandInfo
from mlia.cli.helpers import CLIActionResolver
from mlia.cli.logging import setup_logging
from mlia.cli.options import add_backend_install_options
from mlia.cli.options import add_backend_uninstall_options
from mlia.cli.options import add_custom_supported_operators_options
from mlia.cli.options import add_debug_options
from mlia.cli.options import add_evaluation_options
from mlia.cli.options import add_keras_model_options
from mlia.cli.options import add_multi_optimization_options
from mlia.cli.options import add_optional_tflite_model_options
from mlia.cli.options import add_output_options
from mlia.cli.options import add_target_options
from mlia.cli.options import add_tflite_model_options
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.core.errors import InternalError


logger = logging.getLogger(__name__)

INFO_MESSAGE = f"""
ML Inference Advisor {__version__}

Help the design and optimization of neural network models for efficient inference on a target CPU and NPU

Supported targets:

 - Cortex-A  <op compatibility>
 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>
 - TOSA      <op compatibility>

""".strip()


def get_commands() -> list[CommandInfo]:
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
                add_evaluation_options,
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
                partial(add_target_options, profiles_to_skip=["tosa", "cortex-a"]),
                add_tflite_model_options,
                add_output_options,
                add_debug_options,
                add_evaluation_options,
            ],
        ),
        CommandInfo(
            optimization,
            ["opt"],
            [
                partial(add_target_options, profiles_to_skip=["tosa", "cortex-a"]),
                add_keras_model_options,
                add_multi_optimization_options,
                add_output_options,
                add_debug_options,
                add_evaluation_options,
            ],
        ),
    ]


def backend_commands() -> list[CommandInfo]:
    """Return commands configuration."""
    return [
        CommandInfo(
            backend_install,
            [],
            [
                add_backend_install_options,
                add_debug_options,
            ],
            name="install",
        ),
        CommandInfo(
            backend_uninstall,
            [],
            [
                add_backend_uninstall_options,
                add_debug_options,
            ],
            name="uninstall",
        ),
        CommandInfo(
            backend_list,
            [],
            [
                add_debug_options,
            ],
            name="list",
        ),
    ]


def get_default_command(commands: list[CommandInfo]) -> str | None:
    """Get name of the default command."""
    marked_as_default = [cmd.command_name for cmd in commands if cmd.is_default]
    assert len(marked_as_default) <= 1, "Only one command could be marked as default"

    return next(iter(marked_as_default), None)


def get_possible_command_names(commands: list[CommandInfo]) -> list[str]:
    """Get all possible command names including aliases."""
    return [
        name_or_alias
        for cmd in commands
        for name_or_alias in cmd.command_name_and_aliases
    ]


def init_commands(
    parser: argparse.ArgumentParser, commands: list[CommandInfo]
) -> argparse.ArgumentParser:
    """Init cli subcommands."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    for command in commands:
        command_parser = subparsers.add_parser(
            command.command_name,
            aliases=command.aliases,
            help=command.command_help,
            allow_abbrev=False,
        )
        command_parser.set_defaults(func=command.func)
        for opt_group in command.opt_groups:
            opt_group(command_parser)

    return parser


def setup_context(
    args: argparse.Namespace, context_var_name: str = "ctx"
) -> tuple[ExecutionContext, dict]:
    """Set up context and resolve function parameters."""
    ctx = ExecutionContext(
        working_dir=args.working_dir,
        verbose="verbose" in args and args.verbose,
        action_resolver=CLIActionResolver(vars(args)),
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
    except InternalError as err:
        logger.error("Internal error: %s", err)
    except ConfigurationError as err:
        logger.error(err)
    except Exception as err:  # pylint: disable=broad-except
        logger.error(
            "\nExecution finished with error: %s",
            err,
            exc_info=err if ctx.verbose else None,
        )

        err_advice_message = (
            f"Please check the log files in the {ctx.logs_path} for more details"
        )
        if not ctx.verbose:
            err_advice_message += ", or enable verbose mode (--verbose)"

        logger.error(err_advice_message)

    return 1


def init_common_parser() -> argparse.ArgumentParser:
    """Init common parser."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
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
        allow_abbrev=False,
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


def add_default_command_if_needed(
    args: list[str], input_commands: list[CommandInfo]
) -> None:
    """Add default command to the list of the arguments if needed."""
    default_command = get_default_command(input_commands)

    if default_command and len(args) > 0:
        commands = get_possible_command_names(input_commands)
        help_or_version = ["-h", "--help", "-v", "--version"]

        command_is_missing = args[0] not in [*commands, *help_or_version]
        if command_is_missing:
            args.insert(0, default_command)


def generic_main(
    commands: list[CommandInfo], argv: list[str] | None = None
) -> argparse.Namespace:
    """Enable multiple entry points."""
    common_parser = init_common_parser()
    subcommand_parser = init_subcommand_parser(common_parser)
    init_commands(subcommand_parser, commands)

    common_args, subcommand_args = common_parser.parse_known_args(argv)

    add_default_command_if_needed(subcommand_args, commands)

    args = subcommand_parser.parse_args(subcommand_args, common_args)
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point of the main application."""
    args = generic_main(get_commands(), argv)
    return run_command(args)


def backend_main(argv: list[str] | None = None) -> int:
    """Entry point of the backend application."""
    args = generic_main(backend_commands(), argv)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
