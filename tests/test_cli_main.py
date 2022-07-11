# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for main module."""
import argparse
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

import mlia
from mlia.cli.main import CommandInfo
from mlia.cli.main import main
from mlia.core.context import ExecutionContext
from tests.utils.logging import clear_loggers


def teardown_function() -> None:
    """Perform action after test completion.

    This function is launched automatically by pytest after each test
    in this module.
    """
    clear_loggers()


def test_option_version(capfd: pytest.CaptureFixture) -> None:
    """Test --version."""
    with pytest.raises(SystemExit) as ex:
        main(["--version"])

    assert ex.type == SystemExit
    assert ex.value.code == 0

    stdout, stderr = capfd.readouterr()
    assert len(stdout.splitlines()) == 1
    assert stderr == ""


@pytest.mark.parametrize(
    "is_default, expected_command_help",
    [(True, "Test command [default]"), (False, "Test command")],
)
def test_command_info(is_default: bool, expected_command_help: str) -> None:
    """Test properties of CommandInfo object."""

    def test_command() -> None:
        """Test command."""

    command_info = CommandInfo(test_command, ["test"], [], is_default)
    assert command_info.command_name == "test_command"
    assert command_info.command_name_and_aliases == ["test_command", "test"]
    assert command_info.command_help == expected_command_help


def test_default_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test adding default command."""

    def mock_command(
        func_mock: MagicMock, name: str, with_working_dir: bool
    ) -> Callable[..., None]:
        """Mock cli command."""

        def sample_cmd_1(*args: Any, **kwargs: Any) -> None:
            """Sample command."""
            func_mock(*args, **kwargs)

        def sample_cmd_2(ctx: ExecutionContext, **kwargs: Any) -> None:
            """Another sample command."""
            func_mock(ctx=ctx, **kwargs)

        ret_func = sample_cmd_2 if with_working_dir else sample_cmd_1
        ret_func.__name__ = name

        return ret_func  # type: ignore

    default_command = MagicMock()
    non_default_command = MagicMock()

    def default_command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for default command."""
        parser.add_argument("--sample")
        parser.add_argument("--default_arg", default="123")

    def non_default_command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for non default command."""
        parser.add_argument("--param")

    monkeypatch.setattr(
        "mlia.cli.main.get_commands",
        MagicMock(
            return_value=[
                CommandInfo(
                    func=mock_command(default_command, "default_command", True),
                    aliases=["command1"],
                    opt_groups=[default_command_params],
                    is_default=True,
                ),
                CommandInfo(
                    func=mock_command(
                        non_default_command, "non_default_command", False
                    ),
                    aliases=["command2"],
                    opt_groups=[non_default_command_params],
                    is_default=False,
                ),
            ]
        ),
    )

    tmp_working_dir = str(tmp_path)
    main(["--working-dir", tmp_working_dir, "--sample", "1"])
    main(["command2", "--param", "test"])

    default_command.assert_called_once_with(ctx=ANY, sample="1", default_arg="123")
    non_default_command.assert_called_once_with(param="test")


@pytest.mark.parametrize(
    "params, expected_call",
    [
        [
            ["operators", "sample_model.tflite"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.tflite",
                output=None,
                supported_ops_report=False,
            ),
        ],
        [
            ["ops", "sample_model.tflite"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.tflite",
                output=None,
                supported_ops_report=False,
            ),
        ],
        [
            ["operators", "sample_model.tflite", "--target-profile", "ethos-u55-128"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-128",
                model="sample_model.tflite",
                output=None,
                supported_ops_report=False,
            ),
        ],
        [
            ["operators"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model=None,
                output=None,
                supported_ops_report=False,
            ),
        ],
        [
            ["operators", "--supported-ops-report"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model=None,
                output=None,
                supported_ops_report=True,
            ),
        ],
        [
            [
                "all_tests",
                "sample_model.h5",
                "--optimization-type",
                "pruning",
                "--optimization-target",
                "0.5",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                optimization_type="pruning",
                optimization_target="0.5",
                output=None,
                evaluate_on=["Vela"],
            ),
        ],
        [
            ["sample_model.h5"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                optimization_type="pruning,clustering",
                optimization_target="0.5,32",
                output=None,
                evaluate_on=["Vela"],
            ),
        ],
        [
            ["performance", "sample_model.h5", "--output", "result.json"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                output="result.json",
                evaluate_on=["Vela"],
            ),
        ],
        [
            ["perf", "sample_model.h5", "--target-profile", "ethos-u55-128"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-128",
                model="sample_model.h5",
                output=None,
                evaluate_on=["Vela"],
            ),
        ],
        [
            ["optimization", "sample_model.h5"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                optimization_type="pruning,clustering",
                optimization_target="0.5,32",
                output=None,
                evaluate_on=["Vela"],
            ),
        ],
        [
            ["optimization", "sample_model.h5", "--evaluate-on", "some_backend"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                optimization_type="pruning,clustering",
                optimization_target="0.5,32",
                output=None,
                evaluate_on=["some_backend"],
            ),
        ],
    ],
)
def test_commands_execution(
    monkeypatch: pytest.MonkeyPatch, params: List[str], expected_call: Any
) -> None:
    """Test calling commands from the main function."""
    mock = MagicMock()

    def wrap_mock_command(command: Callable) -> Callable:
        """Wrap the command with the mock."""

        @wraps(command)
        def mock_command(*args: Any, **kwargs: Any) -> Any:
            """Mock the command."""
            mock(*args, **kwargs)

        return mock_command

    monkeypatch.setattr(
        "mlia.cli.options.get_default_backends", MagicMock(return_value=["Vela"])
    )

    monkeypatch.setattr(
        "mlia.cli.options.get_available_backends",
        MagicMock(return_value=["Vela", "some_backend"]),
    )

    for command in ["all_tests", "operators", "performance", "optimization"]:
        monkeypatch.setattr(
            f"mlia.cli.main.{command}",
            wrap_mock_command(getattr(mlia.cli.main, command)),
        )

    main(params)

    mock.assert_called_once_with(*expected_call.args, **expected_call.kwargs)


@pytest.mark.parametrize(
    "verbose, exc_mock, expected_output",
    [
        [
            True,
            MagicMock(side_effect=Exception("Error")),
            [
                "Execution finished with error: Error",
                f"Please check the log files in the {Path.cwd()/'mlia_output/logs'} "
                "for more details",
            ],
        ],
        [
            False,
            MagicMock(side_effect=Exception("Error")),
            [
                "Execution finished with error: Error",
                f"Please check the log files in the {Path.cwd()/'mlia_output/logs'} "
                "for more details, or enable verbose mode",
            ],
        ],
        [
            False,
            MagicMock(side_effect=KeyboardInterrupt()),
            ["Execution has been interrupted"],
        ],
    ],
)
def test_verbose_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    verbose: bool,
    exc_mock: MagicMock,
    expected_output: List[str],
) -> None:
    """Test flag --verbose."""

    def command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for non default command."""
        parser.add_argument("--verbose", action="store_true")

    def command() -> None:
        """Run test command."""
        exc_mock()

    monkeypatch.setattr(
        "mlia.cli.main.get_commands",
        MagicMock(
            return_value=[
                CommandInfo(
                    func=command,
                    aliases=["command"],
                    opt_groups=[command_params],
                ),
            ]
        ),
    )

    params = ["command"]
    if verbose:
        params.append("--verbose")

    exit_code = main(params)
    assert exit_code == 1

    stdout, _ = capsys.readouterr()
    for expected_message in expected_output:
        assert expected_message in stdout
