# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module to manage the CLI interface of systems."""
import json
from pathlib import Path
from typing import cast

import click

from aiet.backend.application import get_available_applications
from aiet.backend.system import get_available_systems
from aiet.backend.system import get_available_systems_directory_names
from aiet.backend.system import get_system
from aiet.backend.system import install_system
from aiet.backend.system import remove_system
from aiet.backend.system import System
from aiet.cli.common import get_format
from aiet.cli.common import print_command_details
from aiet.cli.common import set_format


@click.group(name="system")
@click.option(
    "-f",
    "--format",
    "format_",
    type=click.Choice(["cli", "json"]),
    default="cli",
    show_default=True,
)
@click.pass_context
def system_cmd(ctx: click.Context, format_: str) -> None:
    """Sub command to manage systems."""
    set_format(ctx, format_)


@system_cmd.command(name="list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List all available systems."""
    available_systems = get_available_systems()
    system_names = [system.name for system in available_systems]
    if get_format(ctx) == "json":
        data = {"type": "system", "available": system_names}
        print(json.dumps(data))
    else:
        print("Available systems:\n")
        print(*system_names, sep="\n")


@system_cmd.command(name="details")
@click.option(
    "-n",
    "--name",
    "system_name",
    type=click.Choice([s.name for s in get_available_systems()]),
    required=True,
)
@click.pass_context
def details_cmd(ctx: click.Context, system_name: str) -> None:
    """Details of a specific system."""
    system = cast(System, get_system(system_name))
    applications = [
        s.name for s in get_available_applications() if s.can_run_on(system.name)
    ]
    system_details = system.get_details()
    if get_format(ctx) == "json":
        system_details["available_application"] = applications
        print(json.dumps(system_details))
    else:
        system_details_template = (
            'System "{name}" details\n'
            "Description: {description}\n"
            "Data Transfer Protocol: {protocol}\n"
            "Available Applications: {available_application}"
        )
        print(
            system_details_template.format(
                name=system_details["name"],
                description=system_details["description"],
                protocol=system_details["data_transfer_protocol"],
                available_application=", ".join(applications),
            )
        )

        if system_details["annotations"]:
            print("Annotations:")
            for ann_name, ann_value in system_details["annotations"].items():
                print("\t{}: {}".format(ann_name, ann_value))

        command_details = system_details["commands"]
        for command, details in command_details.items():
            print("\n{} commands:".format(command))
            print_command_details(details)


@system_cmd.command(name="install")
@click.option(
    "-s",
    "--source",
    "source",
    required=True,
    help="Path to the directory or archive with system definition",
)
def install_cmd(source: str) -> None:
    """Install new system."""
    source_path = Path(source)
    install_system(source_path)


@system_cmd.command(name="remove")
@click.option(
    "-d",
    "--directory_name",
    "directory_name",
    type=click.Choice(get_available_systems_directory_names()),
    required=True,
    help="Name of the directory with system",
)
def remove_cmd(directory_name: str) -> None:
    """Remove system by given name."""
    remove_system(directory_name)
