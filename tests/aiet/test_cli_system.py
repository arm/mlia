# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for testing CLI system subcommand."""
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from aiet.backend.config import SystemConfig
from aiet.backend.system import load_system
from aiet.backend.system import System
from aiet.cli.system import details_cmd
from aiet.cli.system import install_cmd
from aiet.cli.system import list_cmd
from aiet.cli.system import remove_cmd
from aiet.cli.system import system_cmd


def test_system_cmd() -> None:
    """Test system commands."""
    commands = ["list", "details", "install", "remove"]
    assert all(command in system_cmd.commands for command in commands)


@pytest.mark.parametrize("format_", ["json", "cli"])
def test_system_cmd_context(cli_runner: CliRunner, format_: str) -> None:
    """Test setting command context parameters."""
    result = cli_runner.invoke(system_cmd, ["--format", format_])
    # command should fail if no subcommand provided
    assert result.exit_code == 2

    result = cli_runner.invoke(system_cmd, ["--format", format_, "list"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "format_,expected_output",
    [
        ("json", '{"type": "system", "available": ["system1", "system2"]}\n'),
        ("cli", "Available systems:\n\nsystem1\nsystem2\n"),
    ],
)
def test_list_cmd_with_format(
    cli_runner: CliRunner, monkeypatch: Any, format_: str, expected_output: str
) -> None:
    """Test available systems command with different formats output."""
    # Mock some systems
    mock_system1 = MagicMock()
    mock_system1.name = "system1"
    mock_system2 = MagicMock()
    mock_system2.name = "system2"

    # Monkey patch the call get_available_systems
    mock_available_systems = MagicMock()
    mock_available_systems.return_value = [mock_system1, mock_system2]
    monkeypatch.setattr("aiet.cli.system.get_available_systems", mock_available_systems)

    obj = {"format": format_}
    result = cli_runner.invoke(list_cmd, obj=obj)
    assert result.output == expected_output


def get_test_system(
    annotations: Optional[Dict[str, Union[str, List[str]]]] = None
) -> System:
    """Return test system details."""
    config = SystemConfig(
        name="system",
        description="test",
        data_transfer={
            "protocol": "ssh",
            "username": "root",
            "password": "root",
            "hostname": "localhost",
            "port": "8022",
        },
        commands={
            "clean": ["clean"],
            "build": ["build"],
            "run": ["run"],
            "post_run": ["post_run"],
        },
        annotations=annotations or {},
    )

    return load_system(config)


def get_details_cmd_json_output(
    annotations: Optional[Dict[str, Union[str, List[str]]]] = None
) -> str:
    """Test JSON output for details command."""
    ann_str = ""
    if annotations is not None:
        ann_str = '"annotations":{},'.format(json.dumps(annotations))

    json_output = (
        """
{
  "type": "system",
  "name": "system",
  "description": "test",
  "data_transfer_protocol": "ssh",
  "commands": {
    "clean":
      {
        "command_strings": ["clean"],
        "user_params": []
      },
    "build":
      {
        "command_strings": ["build"],
        "user_params": []
      },
    "run":
      {
        "command_strings": ["run"],
        "user_params": []
      },
    "post_run":
      {
        "command_strings": ["post_run"],
        "user_params": []
      }
  },
"""
        + ann_str
        + """
  "available_application" : []
  }
"""
    )
    return json.dumps(json.loads(json_output)) + "\n"


def get_details_cmd_console_output(
    annotations: Optional[Dict[str, Union[str, List[str]]]] = None
) -> str:
    """Test console output for details command."""
    ann_str = ""
    if annotations:
        val_str = "".join(
            "\n\t{}: {}".format(ann_name, ann_value)
            for ann_name, ann_value in annotations.items()
        )
        ann_str = "\nAnnotations:{}".format(val_str)
    return (
        'System "system" details'
        + "\nDescription: test"
        + "\nData Transfer Protocol: ssh"
        + "\nAvailable Applications: "
        + ann_str
        + "\n\nclean commands:"
        + "\nCommands: ['clean']"
        + "\n\nbuild commands:"
        + "\nCommands: ['build']"
        + "\n\nrun commands:"
        + "\nCommands: ['run']"
        + "\n\npost_run commands:"
        + "\nCommands: ['post_run']"
        + "\n"
    )


@pytest.mark.parametrize(
    "format_,system,expected_output",
    [
        (
            "json",
            get_test_system(annotations={"ann1": "annotation1", "ann2": ["a1", "a2"]}),
            get_details_cmd_json_output(
                annotations={"ann1": "annotation1", "ann2": ["a1", "a2"]}
            ),
        ),
        (
            "cli",
            get_test_system(annotations={"ann1": "annotation1", "ann2": ["a1", "a2"]}),
            get_details_cmd_console_output(
                annotations={"ann1": "annotation1", "ann2": ["a1", "a2"]}
            ),
        ),
        (
            "json",
            get_test_system(annotations={}),
            get_details_cmd_json_output(annotations={}),
        ),
        (
            "cli",
            get_test_system(annotations={}),
            get_details_cmd_console_output(annotations={}),
        ),
    ],
)
def test_details_cmd(
    cli_runner: CliRunner,
    monkeypatch: Any,
    format_: str,
    system: System,
    expected_output: str,
) -> None:
    """Test details command with different formats output."""
    mock_get_system = MagicMock()
    mock_get_system.return_value = system
    monkeypatch.setattr("aiet.cli.system.get_system", mock_get_system)

    args = ["--name", "system"]
    obj = {"format": format_}
    details_cmd.params[0].type = click.Choice(["system"])

    result = cli_runner.invoke(details_cmd, args=args, obj=obj)
    assert result.output == expected_output


def test_install_cmd(cli_runner: CliRunner, monkeypatch: Any) -> None:
    """Test install system command."""
    mock_install_system = MagicMock()
    monkeypatch.setattr("aiet.cli.system.install_system", mock_install_system)

    args = ["--source", "test"]
    cli_runner.invoke(install_cmd, args=args)
    mock_install_system.assert_called_once_with(Path("test"))


def test_remove_cmd(cli_runner: CliRunner, monkeypatch: Any) -> None:
    """Test remove system command."""
    mock_remove_system = MagicMock()
    monkeypatch.setattr("aiet.cli.system.remove_system", mock_remove_system)
    remove_cmd.params[0].type = click.Choice(["test"])

    args = ["--directory_name", "test"]
    cli_runner.invoke(remove_cmd, args=args)
    mock_remove_system.assert_called_once_with("test")
