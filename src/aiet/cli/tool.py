# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module to manage the CLI interface of tools."""
import json
from typing import Any
from typing import List
from typing import Optional

import click

from aiet.backend.execution import execute_tool_command
from aiet.backend.tool import get_tool
from aiet.backend.tool import get_unique_tool_names
from aiet.cli.common import get_format
from aiet.cli.common import middleware_exception_handler
from aiet.cli.common import middleware_signal_handler
from aiet.cli.common import print_command_details
from aiet.cli.common import set_format


@click.group(name="tool")
@click.option(
    "-f",
    "--format",
    "format_",
    type=click.Choice(["cli", "json"]),
    default="cli",
    show_default=True,
)
@click.pass_context
def tool_cmd(ctx: click.Context, format_: str) -> None:
    """Sub command to manage tools."""
    set_format(ctx, format_)


@tool_cmd.command(name="list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List all available tools."""
    # raise NotImplementedError("TODO")
    tool_names = get_unique_tool_names()
    tool_names.sort()
    if get_format(ctx) == "json":
        data = {"type": "tool", "available": tool_names}
        print(json.dumps(data))
    else:
        print("Available tools:\n")
        print(*tool_names, sep="\n")


def validate_system(
    ctx: click.Context,
    _: click.Parameter,  # param is not used
    value: Any,
) -> Any:
    """Validate provided system name depending on the the tool name."""
    tool_name = ctx.params["tool_name"]
    tools = get_tool(tool_name, value)
    if not tools:
        supported_systems = [tool.supported_systems[0] for tool in get_tool(tool_name)]
        raise click.BadParameter(
            message="'{}' is not one of {}.".format(
                value,
                ", ".join("'{}'".format(system) for system in supported_systems),
            ),
            ctx=ctx,
        )
    return value


@tool_cmd.command(name="details")
@click.option(
    "-n",
    "--name",
    "tool_name",
    type=click.Choice(get_unique_tool_names()),
    required=True,
)
@click.option(
    "-s",
    "--system",
    "system_name",
    callback=validate_system,
    required=False,
)
@click.pass_context
@middleware_signal_handler
@middleware_exception_handler
def details_cmd(ctx: click.Context, tool_name: str, system_name: Optional[str]) -> None:
    """Details of a specific tool."""
    tools = get_tool(tool_name, system_name)
    if get_format(ctx) == "json":
        tools_details = [s.get_details() for s in tools]
        print(json.dumps(tools_details))
    else:
        for tool in tools:
            tool_details = tool.get_details()
            tool_details_template = 'Tool "{name}" details\nDescription: {description}'

            print(
                tool_details_template.format(
                    name=tool_details["name"],
                    description=tool_details["description"],
                )
            )

            print(
                "\nSupported systems: {}".format(
                    ", ".join(tool_details["supported_systems"])
                )
            )

            command_details = tool_details["commands"]

            for command, details in command_details.items():
                print("\n{} commands:".format(command))
                print_command_details(details)


# pylint: disable=too-many-arguments
@tool_cmd.command(name="execute")
@click.option(
    "-n",
    "--name",
    "tool_name",
    type=click.Choice(get_unique_tool_names()),
    required=True,
)
@click.option("-p", "--param", "tool_params", multiple=True)
@click.option(
    "-s",
    "--system",
    "system_name",
    callback=validate_system,
    required=False,
)
@middleware_signal_handler
@middleware_exception_handler
def execute_cmd(
    tool_name: str, tool_params: List[str], system_name: Optional[str]
) -> None:
    """Execute tool commands."""
    execute_tool_command(tool_name, tool_params, system_name)
