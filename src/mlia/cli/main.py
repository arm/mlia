# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI main entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from inspect import signature

from mlia import __version__
from mlia.backend.errors import BackendUnavailableError
from mlia.cli.commands import backend_install
from mlia.cli.commands import backend_list
from mlia.cli.commands import backend_uninstall
from mlia.cli.commands import check
from mlia.cli.commands import optimize
from mlia.cli.common import CommandInfo
from mlia.cli.helpers import CLIActionResolver
from mlia.cli.helpers import copy_profile_file_to_output_dir
from mlia.cli.options import add_backend_install_options
from mlia.cli.options import add_backend_options
from mlia.cli.options import add_backend_uninstall_options
from mlia.cli.options import add_check_category_options
from mlia.cli.options import add_dataset_options
from mlia.cli.options import add_debug_options
from mlia.cli.options import add_keras_model_options
from mlia.cli.options import add_model_options
from mlia.cli.options import add_multi_optimization_options
from mlia.cli.options import add_output_directory
from mlia.cli.options import add_output_options
from mlia.cli.options import add_target_options
from mlia.cli.options import get_output_format
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.core.errors import InternalError
from mlia.core.logging import setup_logging
from mlia.target.registry import table as target_table


logger = logging.getLogger(__name__)

INFO_MESSAGE = f"""
ML Inference Advisor {__version__}

Help the design and optimization of neural network models for efficient inference on a target CPU or NPU.

{target_table().to_plain_text(show_title=True, space=False)}
Use command 'mlia-backend' to install backends.
""".strip()


def get_commands() -> list[CommandInfo]:
    """Return commands configuration."""
    return [
        CommandInfo(
            check,
            [],
            [
                add_output_directory,
                add_model_options,
                partial(
                    add_target_options,
                    supported_advice=[
                        AdviceCategory.COMPATIBILITY,
                        AdviceCategory.PERFORMANCE,
                    ],
                ),
                add_backend_options,
                add_check_category_options,
                add_output_options,
                add_debug_options,
            ],
        ),
        CommandInfo(
            optimize,
            [],
            [
                add_output_directory,
                add_keras_model_options,
                partial(
                    add_target_options, supported_advice=[AdviceCategory.OPTIMIZATION]
                ),
                partial(
                    add_backend_options,
                    backends_to_skip=["tosa-checker", "armnn-tflite-delegate"],
                ),
                add_multi_optimization_options,
                add_output_options,
                add_debug_options,
                add_dataset_options,
            ],
        ),
    ]


def get_backend_commands() -> list[CommandInfo]:
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
    try:
        ctx = ExecutionContext(
            verbose="debug" in args and args.debug,
            action_resolver=CLIActionResolver(vars(args)),
            output_format=get_output_format(args),
            output_dir=args.output_dir if "output_dir" in args else None,
        )
    except Exception as err:  # pylint: disable=broad-except
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    setup_logging(ctx.logs_path, ctx.verbose, ctx.output_format)

    # these parameters should not be passed into command function
    skipped_params = ["func", "command", "debug", "json", "output_dir"]

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

    logger.debug(
        "*** This is the beginning of the command '%s' execution ***", args.command
    )

    try:
        logger.info("ML Inference Advisor %s", __version__)
        if copy_profile_file(ctx, func_args, "target_profile"):
            logger.info(
                "\nThe target profile (.toml) is copied to the output directory: %s",
                ctx.output_dir,
            )
        if copy_profile_file(ctx, func_args, "optimization_profile"):
            logger.info(
                "\nThe optimization profile (.toml) is copied to "
                "the output directory: %s",
                ctx.output_dir,
            )
        args.func(**func_args)
        return 0
    except KeyboardInterrupt:
        logger.error("Execution has been interrupted")
    except InternalError as err:
        logger.error("Internal error: %s", err)
    except ConfigurationError as err:
        logger.error(err)
    except BackendUnavailableError as err:
        logger.error("Error: Backend %s is not available.", err.backend)
        # Show installation instructions for optional backends
        if err.backend in ("tosa-checker", "vela"):
            logger.error(
                'Please use next command to install it: mlia-backend install "%s"',
                err.backend,
            )
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
            err_advice_message += ", or enable debug mode (--debug)"

        logger.error(err_advice_message)
    finally:
        logger.info("This execution of MLIA used output directory: %s", ctx.output_dir)
    return 1


def init_parser(commands: list[CommandInfo]) -> argparse.ArgumentParser:
    """Init subcommand parser."""
    parser = argparse.ArgumentParser(
        description=INFO_MESSAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    init_commands(parser, commands)
    return parser


def init_and_run(commands: list[CommandInfo], argv: list[str] | None = None) -> int:
    """Init parser and run subcommand."""
    parser = init_parser(commands)
    args = parser.parse_args(argv)

    return run_command(args)


def copy_profile_file(
    ctx: ExecutionContext, func_args: dict, profile_to_copy: str
) -> bool:
    """If present, copy the selected profile file to the output directory."""
    if func_args.get(profile_to_copy):
        return copy_profile_file_to_output_dir(
            func_args[profile_to_copy], ctx.output_dir, profile_to_copy
        )

    return False


def main(argv: list[str] | None = None) -> int:
    """Entry point of the main application."""
    commands = get_commands()
    return init_and_run(commands, argv)


def backend_main(argv: list[str] | None = None) -> int:
    """Entry point of the backend application."""
    commands = get_backend_commands()
    return init_and_run(commands, argv)


if __name__ == "__main__":
    sys.exit(main())
