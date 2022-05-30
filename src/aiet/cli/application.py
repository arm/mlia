# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-FileCopyrightText: Copyright (c) 2021, Gianluca Gippetto. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""Module to manage the CLI interface of applications."""
import json
import logging
import re
from pathlib import Path
from typing import Any
from typing import IO
from typing import List
from typing import Optional
from typing import Tuple

import click
import cloup

from aiet.backend.application import get_application
from aiet.backend.application import get_available_application_directory_names
from aiet.backend.application import get_unique_application_names
from aiet.backend.application import install_application
from aiet.backend.application import remove_application
from aiet.backend.common import DataPaths
from aiet.backend.execution import execute_application_command
from aiet.backend.execution import run_application
from aiet.backend.system import get_available_systems
from aiet.cli.common import get_format
from aiet.cli.common import middleware_exception_handler
from aiet.cli.common import middleware_signal_handler
from aiet.cli.common import print_command_details
from aiet.cli.common import set_format


@click.group(name="application")
@click.option(
    "-f",
    "--format",
    "format_",
    type=click.Choice(["cli", "json"]),
    default="cli",
    show_default=True,
)
@click.pass_context
def application_cmd(ctx: click.Context, format_: str) -> None:
    """Sub command to manage applications."""
    set_format(ctx, format_)


@application_cmd.command(name="list")
@click.pass_context
@click.option(
    "-s",
    "--system",
    "system_name",
    type=click.Choice([s.name for s in get_available_systems()]),
    required=False,
)
def list_cmd(ctx: click.Context, system_name: str) -> None:
    """List all available applications."""
    unique_application_names = get_unique_application_names(system_name)
    unique_application_names.sort()
    if get_format(ctx) == "json":
        data = {"type": "application", "available": unique_application_names}
        print(json.dumps(data))
    else:
        print("Available applications:\n")
        print(*unique_application_names, sep="\n")


@application_cmd.command(name="details")
@click.option(
    "-n",
    "--name",
    "application_name",
    type=click.Choice(get_unique_application_names()),
    required=True,
)
@click.option(
    "-s",
    "--system",
    "system_name",
    type=click.Choice([s.name for s in get_available_systems()]),
    required=False,
)
@click.pass_context
def details_cmd(ctx: click.Context, application_name: str, system_name: str) -> None:
    """Details of a specific application."""
    applications = get_application(application_name, system_name)
    if not applications:
        raise click.UsageError(
            "Application '{}' doesn't support the system '{}'".format(
                application_name, system_name
            )
        )

    if get_format(ctx) == "json":
        applications_details = [s.get_details() for s in applications]
        print(json.dumps(applications_details))
    else:
        for application in applications:
            application_details = application.get_details()
            application_details_template = (
                'Application "{name}" details\nDescription: {description}'
            )

            print(
                application_details_template.format(
                    name=application_details["name"],
                    description=application_details["description"],
                )
            )

            print(
                "\nSupported systems: {}".format(
                    ", ".join(application_details["supported_systems"])
                )
            )

            command_details = application_details["commands"]

            for command, details in command_details.items():
                print("\n{} commands:".format(command))
                print_command_details(details)


# pylint: disable=too-many-arguments
@application_cmd.command(name="execute")
@click.option(
    "-n",
    "--name",
    "application_name",
    type=click.Choice(get_unique_application_names()),
    required=True,
)
@click.option(
    "-s",
    "--system",
    "system_name",
    type=click.Choice([s.name for s in get_available_systems()]),
    required=True,
)
@click.option(
    "-c",
    "--command",
    "command_name",
    type=click.Choice(["build", "run"]),
    required=True,
)
@click.option("-p", "--param", "application_params", multiple=True)
@click.option("--system-param", "system_params", multiple=True)
@click.option("-d", "--deploy", "deploy_params", multiple=True)
@middleware_signal_handler
@middleware_exception_handler
def execute_cmd(
    application_name: str,
    system_name: str,
    command_name: str,
    application_params: List[str],
    system_params: List[str],
    deploy_params: List[str],
) -> None:
    """Execute application commands. DEPRECATED! Use 'aiet application run' instead."""
    logging.warning(
        "Please use 'aiet application run' instead. Use of 'aiet application "
        "execute' is deprecated and might be removed in a future release."
    )

    custom_deploy_data = get_custom_deploy_data(command_name, deploy_params)

    execute_application_command(
        command_name,
        application_name,
        application_params,
        system_name,
        system_params,
        custom_deploy_data,
    )


@cloup.command(name="run")
@cloup.option(
    "-n",
    "--name",
    "application_name",
    type=click.Choice(get_unique_application_names()),
)
@cloup.option(
    "-s",
    "--system",
    "system_name",
    type=click.Choice([s.name for s in get_available_systems()]),
)
@cloup.option("-p", "--param", "application_params", multiple=True)
@cloup.option("--system-param", "system_params", multiple=True)
@cloup.option("-d", "--deploy", "deploy_params", multiple=True)
@click.option(
    "-r",
    "--report",
    "report_file",
    type=Path,
    help="Create a report file in JSON format containing metrics parsed from "
    "the simulation output as specified in the aiet-config.json.",
)
@cloup.option(
    "--config",
    "config_file",
    type=click.File("r"),
    help="Read options from a config file rather than from the command line. "
    "The config file is a json file.",
)
@cloup.constraint(
    cloup.constraints.If(
        cloup.constraints.conditions.Not(
            cloup.constraints.conditions.IsSet("config_file")
        ),
        then=cloup.constraints.require_all,
    ),
    ["system_name", "application_name"],
)
@cloup.constraint(
    cloup.constraints.If("config_file", then=cloup.constraints.accept_none),
    [
        "system_name",
        "application_name",
        "application_params",
        "system_params",
        "deploy_params",
    ],
)
@middleware_signal_handler
@middleware_exception_handler
def run_cmd(
    application_name: str,
    system_name: str,
    application_params: List[str],
    system_params: List[str],
    deploy_params: List[str],
    report_file: Optional[Path],
    config_file: Optional[IO[str]],
) -> None:
    """Execute application commands."""
    if config_file:
        payload_data = json.load(config_file)
        (
            system_name,
            application_name,
            application_params,
            system_params,
            deploy_params,
            report_file,
        ) = parse_payload_run_config(payload_data)

    custom_deploy_data = get_custom_deploy_data("run", deploy_params)

    run_application(
        application_name,
        application_params,
        system_name,
        system_params,
        custom_deploy_data,
        report_file,
    )


application_cmd.add_command(run_cmd)


def parse_payload_run_config(
    payload_data: dict,
) -> Tuple[str, str, List[str], List[str], List[str], Optional[Path]]:
    """Parse the payload into a tuple."""
    system_id = payload_data.get("id")
    arguments: Optional[Any] = payload_data.get("arguments")

    if not isinstance(system_id, str):
        raise click.ClickException("invalid payload json: no system 'id'")
    if not isinstance(arguments, dict):
        raise click.ClickException("invalid payload json: no arguments object")

    application_name = arguments.pop("application", None)
    if not isinstance(application_name, str):
        raise click.ClickException("invalid payload json: no application_id")

    report_path = arguments.pop("report_path", None)

    application_params = []
    system_params = []
    deploy_params = []

    for (param_key, value) in arguments.items():
        (par, _) = re.subn("^application/", "", param_key)
        (par, found_sys_param) = re.subn("^system/", "", par)
        (par, found_deploy_param) = re.subn("^deploy/", "", par)

        param_expr = par + "=" + value
        if found_sys_param:
            system_params.append(param_expr)
        elif found_deploy_param:
            deploy_params.append(par)
        else:
            application_params.append(param_expr)

    return (
        system_id,
        application_name,
        application_params,
        system_params,
        deploy_params,
        report_path,
    )


def get_custom_deploy_data(
    command_name: str, deploy_params: List[str]
) -> List[DataPaths]:
    """Get custom deploy data information."""
    custom_deploy_data: List[DataPaths] = []
    if not deploy_params:
        return custom_deploy_data

    for param in deploy_params:
        parts = param.split(":")
        if not len(parts) == 2 or any(not part.strip() for part in parts):
            raise click.ClickException(
                "Invalid deploy parameter '{}' for command {}".format(
                    param, command_name
                )
            )
        data_path = DataPaths(Path(parts[0]), parts[1])
        if not data_path.src.exists():
            raise click.ClickException("Path {} does not exist".format(data_path.src))
        custom_deploy_data.append(data_path)

    return custom_deploy_data


@application_cmd.command(name="install")
@click.option(
    "-s",
    "--source",
    "source",
    required=True,
    help="Path to the directory or archive with application definition",
)
def install_cmd(source: str) -> None:
    """Install new application."""
    source_path = Path(source)
    install_application(source_path)


@application_cmd.command(name="remove")
@click.option(
    "-d",
    "--directory_name",
    "directory_name",
    type=click.Choice(get_available_application_directory_names()),
    required=True,
    help="Name of the directory with application",
)
def remove_cmd(directory_name: str) -> None:
    """Remove application."""
    remove_application(directory_name)
