# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=attribute-defined-outside-init,no-member,line-too-long,too-many-arguments,too-many-locals
"""Module for testing CLI tool subcommand."""
import json
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
from click.testing import Result

from aiet.backend.tool import get_unique_tool_names
from aiet.backend.tool import Tool
from aiet.cli.tool import details_cmd
from aiet.cli.tool import execute_cmd
from aiet.cli.tool import list_cmd
from aiet.cli.tool import tool_cmd


def test_tool_cmd() -> None:
    """Test tool commands."""
    commands = ["list", "details", "execute"]
    assert all(command in tool_cmd.commands for command in commands)


@pytest.mark.parametrize("format_", ["json", "cli"])
def test_tool_cmd_context(cli_runner: CliRunner, format_: str) -> None:
    """Test setting command context parameters."""
    result = cli_runner.invoke(tool_cmd, ["--format", format_])
    # command should fail if no subcommand provided
    assert result.exit_code == 2

    result = cli_runner.invoke(tool_cmd, ["--format", format_, "list"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "format_, expected_output",
    [
        (
            "json",
            '{"type": "tool", "available": ["tool_1", "tool_2"]}\n',
        ),
        ("cli", "Available tools:\n\ntool_1\ntool_2\n"),
    ],
)
def test_list_cmd(
    cli_runner: CliRunner,
    monkeypatch: Any,
    format_: str,
    expected_output: str,
) -> None:
    """Test available tool commands."""
    # Mock some tools
    mock_tool_1 = MagicMock(spec=Tool)
    mock_tool_1.name = "tool_1"
    mock_tool_2 = MagicMock(spec=Tool)
    mock_tool_2.name = "tool_2"

    # Monkey patch the call get_available_tools
    mock_available_tools = MagicMock()
    mock_available_tools.return_value = [mock_tool_1, mock_tool_2]

    monkeypatch.setattr("aiet.backend.tool.get_available_tools", mock_available_tools)

    obj = {"format": format_}
    args: Sequence[str] = []
    result = cli_runner.invoke(list_cmd, obj=obj, args=args)
    assert result.output == expected_output


def get_details_cmd_json_output() -> List[dict]:
    """Get JSON output for details command."""
    json_output = [
        {
            "type": "tool",
            "name": "tool_1",
            "description": "This is tool 1",
            "supported_systems": ["System 1"],
            "commands": {
                "clean": {"command_strings": ["echo 'clean'"], "user_params": []},
                "build": {"command_strings": ["echo 'build'"], "user_params": []},
                "run": {"command_strings": ["echo 'run'"], "user_params": []},
                "post_run": {"command_strings": ["echo 'post_run'"], "user_params": []},
            },
        }
    ]

    return json_output


def get_details_cmd_console_output() -> str:
    """Get console output for details command."""
    return (
        'Tool "tool_1" details'
        "\nDescription: This is tool 1"
        "\n\nSupported systems: System 1"
        "\n\nclean commands:"
        "\nCommands: [\"echo 'clean'\"]"
        "\n\nbuild commands:"
        "\nCommands: [\"echo 'build'\"]"
        "\n\nrun commands:\nCommands: [\"echo 'run'\"]"
        "\n\npost_run commands:"
        "\nCommands: [\"echo 'post_run'\"]"
        "\n"
    )


@pytest.mark.parametrize(
    [
        "tool_name",
        "format_",
        "expected_success",
        "expected_output",
    ],
    [
        ("tool_1", "json", True, get_details_cmd_json_output()),
        ("tool_1", "cli", True, get_details_cmd_console_output()),
        ("non-existent tool", "json", False, None),
        ("non-existent tool", "cli", False, None),
    ],
)
def test_details_cmd(
    cli_runner: CliRunner,
    tool_name: str,
    format_: str,
    expected_success: bool,
    expected_output: str,
) -> None:
    """Test tool details command."""
    details_cmd.params[0].type = click.Choice(["tool_1", "tool_2", "vela"])
    result = cli_runner.invoke(
        details_cmd, obj={"format": format_}, args=["--name", tool_name]
    )
    success = result.exit_code == 0
    assert success == expected_success, result.output
    if expected_success:
        assert result.exception is None
        output = json.loads(result.output) if format_ == "json" else result.output
        assert output == expected_output


@pytest.mark.parametrize(
    "system_name",
    [
        "",
        "Corstone-300: Cortex-M55+Ethos-U55",
        "Corstone-300: Cortex-M55+Ethos-U65",
        "Corstone-310: Cortex-M85+Ethos-U55",
    ],
)
def test_details_cmd_vela(cli_runner: CliRunner, system_name: str) -> None:
    """Test tool details command for Vela."""
    details_cmd.params[0].type = click.Choice(get_unique_tool_names())
    details_cmd.params[1].type = click.Choice([system_name])
    args = ["--name", "vela"]
    if system_name:
        args += ["--system", system_name]
    result = cli_runner.invoke(details_cmd, obj={"format": "json"}, args=args)
    success = result.exit_code == 0
    assert success, result.output
    result_json = json.loads(result.output)
    assert result_json
    if system_name:
        assert len(result_json) == 1
        tool = result_json[0]
        assert len(tool["supported_systems"]) == 1
        assert system_name == tool["supported_systems"][0]
    else:  # no system specified => list details for all systems
        assert len(result_json) == 3
        assert all(len(tool["supported_systems"]) == 1 for tool in result_json)


@pytest.fixture(scope="session")
def input_model_file(non_optimised_input_model_file: Path) -> Path:
    """Provide the path to a quantized dummy model file in the test_resources_path."""
    return non_optimised_input_model_file


def execute_vela(
    cli_runner: CliRunner,
    tool_name: str = "vela",
    system_name: Optional[str] = None,
    input_model: Optional[Path] = None,
    output_model: Optional[Path] = None,
    mac: Optional[int] = None,
    format_: str = "cli",
) -> Result:
    """Run Vela with different parameters."""
    execute_cmd.params[0].type = click.Choice(get_unique_tool_names())
    execute_cmd.params[2].type = click.Choice([system_name or "dummy_system"])
    args = ["--name", tool_name]
    if system_name is not None:
        args += ["--system", system_name]
    if input_model is not None:
        args += ["--param", "input={}".format(input_model)]
    if output_model is not None:
        args += ["--param", "output={}".format(output_model)]
    if mac is not None:
        args += ["--param", "mac={}".format(mac)]
    result = cli_runner.invoke(
        execute_cmd,
        args=args,
        obj={"format": format_},
    )
    return result


@pytest.mark.parametrize("format_", ["cli, json"])
@pytest.mark.parametrize(
    ["tool_name", "system_name", "mac", "expected_success", "expected_output"],
    [
        ("vela", "System 1", 32, False, None),  # system not supported
        ("vela", "NON-EXISTENT SYSTEM", 128, False, None),  # system does not exist
        ("vela", "Corstone-300: Cortex-M55+Ethos-U55", 32, True, None),
        ("NON-EXISTENT TOOL", "Corstone-300: Cortex-M55+Ethos-U55", 32, False, None),
        ("vela", "Corstone-300: Cortex-M55+Ethos-U55", 64, True, None),
        ("vela", "Corstone-300: Cortex-M55+Ethos-U55", 128, True, None),
        ("vela", "Corstone-300: Cortex-M55+Ethos-U55", 256, True, None),
        (
            "vela",
            "Corstone-300: Cortex-M55+Ethos-U55",
            512,
            False,
            None,
        ),  # mac not supported
        (
            "vela",
            "Corstone-300: Cortex-M55+Ethos-U65",
            32,
            False,
            None,
        ),  # mac not supported
        ("vela", "Corstone-300: Cortex-M55+Ethos-U65", 256, True, None),
        ("vela", "Corstone-300: Cortex-M55+Ethos-U65", 512, True, None),
        (
            "vela",
            None,
            512,
            False,
            "Error: Please specify the system for tool vela.",
        ),  # no system specified
        (
            "NON-EXISTENT TOOL",
            "Corstone-300: Cortex-M55+Ethos-U65",
            512,
            False,
            None,
        ),  # tool does not exist
        ("vela", "Corstone-310: Cortex-M85+Ethos-U55", 128, True, None),
    ],
)
def test_vela_run(
    cli_runner: CliRunner,
    format_: str,
    input_model_file: Path,  # pylint: disable=redefined-outer-name
    tool_name: str,
    system_name: Optional[str],
    mac: int,
    expected_success: bool,
    expected_output: Optional[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test the execution of the Vela command."""
    monkeypatch.chdir(tmp_path)

    output_file = Path("vela_output.tflite")

    result = execute_vela(
        cli_runner,
        tool_name=tool_name,
        system_name=system_name,
        input_model=input_model_file,
        output_model=output_file,
        mac=mac,
        format_=format_,
    )

    success = result.exit_code == 0
    assert success == expected_success
    if success:
        # Check output file
        output_file = output_file.resolve()
        assert output_file.is_file()
    if expected_output:
        assert result.output.strip() == expected_output


@pytest.mark.parametrize("include_input_model", [True, False])
@pytest.mark.parametrize("include_output_model", [True, False])
@pytest.mark.parametrize("include_mac", [True, False])
def test_vela_run_missing_params(
    cli_runner: CliRunner,
    input_model_file: Path,  # pylint: disable=redefined-outer-name
    include_input_model: bool,
    include_output_model: bool,
    include_mac: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test the execution of the Vela command with missing user parameters."""
    monkeypatch.chdir(tmp_path)

    output_model_file = Path("output_model.tflite")
    system_name = "Corstone-300: Cortex-M55+Ethos-U65"
    mac = 256
    # input_model is a required parameters, but mac and output_model have default values.
    expected_success = include_input_model

    result = execute_vela(
        cli_runner,
        tool_name="vela",
        system_name=system_name,
        input_model=input_model_file if include_input_model else None,
        output_model=output_model_file if include_output_model else None,
        mac=mac if include_mac else None,
    )

    success = result.exit_code == 0
    assert success == expected_success, (
        f"Success is {success}, but expected {expected_success}. "
        f"Included params: ["
        f"input_model={include_input_model}, "
        f"output_model={include_output_model}, "
        f"mac={include_mac}]"
    )
