# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""End to end tests for MLIA CLI."""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess  # nosec
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Generator
from typing import Iterable

import pytest

from mlia.cli.config import get_available_backends
from mlia.cli.main import get_commands
from mlia.cli.main import get_possible_command_names
from mlia.cli.main import init_commands
from mlia.cli.main import init_common_parser
from mlia.cli.main import init_subcommand_parser
from mlia.utils.filesystem import get_supported_profile_names
from mlia.utils.types import is_list_of


pytestmark = pytest.mark.e2e

VALID_COMMANDS = get_possible_command_names(get_commands())


@dataclass
class ExecutionConfiguration:
    """Execution configuration."""

    command: str
    parameters: dict[str, list[list[str]]]

    @classmethod
    def from_dict(cls, exec_info: dict) -> ExecutionConfiguration:
        """Create instance from the dictionary."""
        if not (command := exec_info.get("command")):
            raise Exception("Command is not defined")

        if command not in VALID_COMMANDS:
            raise Exception(f"Unknown command {command}")

        if not (params := exec_info.get("parameters")):
            raise Exception(f"Command {command} should have parameters")

        assert isinstance(params, dict), "Parameters should be a dictionary"
        assert all(
            isinstance(param_group_name, str)
            and is_list_of(param_group_values, list)
            and all(is_list_of(param_list, str) for param_list in param_group_values)
            for param_group_name, param_group_values in params.items()
        ), "Execution configuration should be a dictionary of list of list of strings"

        return cls(command, params)

    @property
    def all_combinations(self) -> Iterable[list[str]]:
        """Generate all command combinations."""
        parameter_groups = self.parameters.values()
        parameter_combinations = itertools.product(*parameter_groups)

        return (
            [self.command, *itertools.chain.from_iterable(param_combination)]
            for param_combination in parameter_combinations
        )


def launch_and_wait(cmd: list[str], stdin: Any | None = None) -> None:
    """Launch command and wait for the completion."""
    with subprocess.Popen(  # nosec
        cmd,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # redirect command stderr to stdout
        universal_newlines=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise Exception("Unable to get process output")

        # redirect output of the process into current process stdout
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if (exit_code := process.poll()) != 0:
            raise Exception(f"Command failed with exit_code {exit_code}")


def run_command(cmd: list[str], cmd_input: str | None = None) -> None:
    """Run command."""
    print(f"Run command: {' '.join(cmd)}")

    with ExitStack() as exit_stack:
        cmd_input_file = None

        if cmd_input is not None:
            print(f"Command will receive next input: {repr(cmd_input)}")

            cmd_input_file = (
                tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
                    mode="w", prefix="mlia_", suffix="_test"
                )
            )
            exit_stack.enter_context(cmd_input_file)

            cmd_input_file.write(cmd_input)
            cmd_input_file.seek(0)

        launch_and_wait(cmd, cmd_input_file)


def get_config_file() -> Path:
    """Get path to the configuration file."""
    env_var_name = "MLIA_E2E_CONFIG_FILE"
    if not (config_file_env_var := os.environ.get(env_var_name)):
        raise Exception(f"Config file env variable ({env_var_name}) is not set")

    config_file = Path(config_file_env_var)
    if not config_file.is_file():
        raise Exception(f"Invalid config file {config_file_env_var}")

    return config_file


def get_args_parser() -> Any:
    """Return MLIA argument parser."""
    common_parser = init_common_parser()
    subcommand_parser = init_subcommand_parser(common_parser)
    init_commands(subcommand_parser, get_commands())

    return subcommand_parser


def replace_element(params: list[str], idx: int, value: str) -> list[str]:
    """Replace element in the list at the index."""
    # fmt: off
    return [*params[:idx], value, *params[idx + 1:]]
    # fmt: on


def resolve(params: list[str]) -> Generator[list[str], None, None]:
    """Replace wildcard with actual param."""
    for idx, param in enumerate(params):
        if "*" not in param:
            continue

        prev = None if idx == 0 else params[idx - 1]

        if prev == "--target-profile" and param == "*":
            resolved = (
                replace_element(params, idx, profile)
                for profile in get_supported_profile_names()
            )
        elif param.startswith("e2e_config") and (
            filenames := glob.glob(f"{Path.cwd()}/{param}", recursive=True)
        ):
            resolved = (
                replace_element(params, idx, filename) for filename in filenames
            )
        else:
            raise ValueError(f"Unable to resolve parameter {param}")

        for item in resolved:
            yield from resolve(item)

        break
    else:
        yield params


def resolve_parameters(executions: dict) -> dict:
    """Resolve command parameters."""
    for execution in executions:
        parameters = execution.get("parameters", {})

        for param_group, param_group_values in parameters.items():
            resolved_params: list[list[str]] = []

            for group in param_group_values:
                if any("*" in item for item in group):
                    resolved_params.extend(resolve(group))
                else:
                    resolved_params.append(group)

            parameters[param_group] = resolved_params

    return executions


def get_config_content(config_file: Path) -> Any:
    """Get executions configuration."""
    with open(config_file, encoding="utf-8") as file:
        json_data = json.load(file)

    assert isinstance(json_data, dict), "JSON configuration expected to be a dictionary"

    executions = json_data.get("executions", [])
    assert is_list_of(executions, dict), "List of the dictionaries expected"

    return executions


def get_all_commands_combinations(executions: Any) -> Generator[list[str], None, None]:
    """Return all commands combinations."""
    exec_configs = (
        ExecutionConfiguration.from_dict(exec_info) for exec_info in executions
    )

    parser = get_args_parser()
    for exec_config in exec_configs:
        for command_combination in exec_config.all_combinations:
            for idx, param in enumerate(command_combination):
                if "{model_name}" in param:
                    args = parser.parse_args(command_combination)
                    model_name = Path(args.model).stem
                    param = param.replace("{model_name}", model_name)
                    command_combination[idx] = param
            yield command_combination


def check_args(args: list[str], no_skip: bool) -> None:
    """Check the arguments and skip/fail test cases based on that."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        help="Backends to use for evaluation.",
        nargs="+",
    )

    parsed_args, _ = parser.parse_known_args(args)
    if parsed_args.backend:
        required_backends = set(parsed_args.backend)
        available_backends = set(get_available_backends())
        missing_backends = required_backends.difference(available_backends)

        if missing_backends and not no_skip:
            pytest.skip(f"Missing backend(s): {','.join(missing_backends)}")


def get_execution_definitions() -> Generator[list[str], None, None]:
    """Collect all execution definitions from configuration file."""
    config_file = get_config_file()
    executions = get_config_content(config_file)
    executions = resolve_parameters(executions)

    return get_all_commands_combinations(executions)


class TestEndToEnd:
    """End to end command tests."""

    @pytest.mark.parametrize("command", get_execution_definitions(), ids=str)
    def test_e2e(self, command: list[str], no_skip: bool) -> None:
        """Test MLIA command with the provided parameters."""
        check_args(command, no_skip)
        mlia_command = ["mlia", *command]
        run_command(mlia_command)
