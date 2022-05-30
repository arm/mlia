# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module to mange the CLI interface."""
import click

from aiet import __version__
from aiet.cli.application import application_cmd
from aiet.cli.completion import completion_cmd
from aiet.cli.system import system_cmd
from aiet.cli.tool import tool_cmd
from aiet.utils.helpers import set_verbosity


@click.group()
@click.version_option(__version__)
@click.option(
    "-v", "--verbose", default=0, count=True, callback=set_verbosity, expose_value=False
)
@click.pass_context
def cli(ctx: click.Context) -> None:  # pylint: disable=unused-argument
    """AIET: AI Evaluation Toolkit."""
    # Unused arguments must be present here in definition to pass click context.


cli.add_command(application_cmd)
cli.add_command(system_cmd)
cli.add_command(tool_cmd)
cli.add_command(completion_cmd)
