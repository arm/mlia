# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Prepare standalone frontend requests through the MLIA CLI declarations."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import copy
from typing import cast

import click
from typer.main import get_command

from mlia.check import CheckError, CheckRequest
from mlia.cli.commands import CheckOptionCommand, mlia_app
from mlia.utils.boundaries.environment import temporary_dotenv_environment


def _enter_context(
    stack: ExitStack,
    command: click.Command,
    name: str,
    parent: click.Context | None = None,
) -> click.Context:
    """Own the context before parsing so failed callbacks release resources."""
    params = []
    for param in command.params:
        if isinstance(param, click.Option) and param.prompt is not None:
            param = copy(param)
            param.prompt = None
        params.append(param)
    command.params = params
    return stack.enter_context(
        command.context_class(
            command, info_name=name, parent=parent, **command.context_settings
        )
    )


@contextmanager
def prepare_check_defaults() -> Iterator[CheckRequest]:
    """Resolve CLI defaults without executing a check or opening the TUI.

    Keep callback resources alive until the caller exits the context. Declarative
    prompts are disabled; required options without defaults or environment values
    raise CheckError. Plugin callbacks must support invocation without terminal
    input. Callback resources retain their existing thread and process limits.
    """
    with temporary_dotenv_environment(), ExitStack() as stack:
        try:
            root = cast(click.Group, get_command(mlia_app))
            root_ctx = _enter_context(stack, root, "mlia")
            remaining = root.parse_args(root_ctx, ["check", "--tui"])
            root_ctx.args = []
            command = cast(CheckOptionCommand, root.get_command(root_ctx, "check"))
            root_ctx.invoked_subcommand = "check"
            click.Command.invoke(root, root_ctx)
            check_ctx = _enter_context(stack, command, "check", root_ctx)
            command.parse_args(check_ctx, remaining)
            request = command.prepare_request(check_ctx)
        except click.ClickException as err:
            raise CheckError(
                f"Cannot prepare CLI defaults: {err.format_message()} "
                "Supply the option with 'mlia check --tui ...'.",
                exit_code=2,
            ) from err
        except (click.Abort, EOFError) as err:
            raise CheckError(
                "A CLI option callback requires terminal input or was aborted. "
                "Configure its option with 'mlia check --tui ...'.",
                exit_code=2,
            ) from err
        yield request
