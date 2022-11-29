# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for main module."""
from __future__ import annotations

import argparse
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable
from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

import mlia
from mlia.backend.errors import BackendUnavailableError
from mlia.cli.main import backend_main
from mlia.cli.main import CommandInfo
from mlia.cli.main import main
from mlia.core.errors import InternalError
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


def test_default_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test adding default command."""

    def mock_command(func_mock: MagicMock, name: str) -> Callable[..., None]:
        """Mock cli command."""

        def sample_cmd_1(*args: Any, **kwargs: Any) -> None:
            """Sample command."""
            func_mock(*args, **kwargs)

        ret_func = sample_cmd_1
        ret_func.__name__ = name

        return ret_func

    non_default_command = MagicMock()

    def non_default_command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for non default command."""
        parser.add_argument("--param")

    monkeypatch.setattr(
        "mlia.cli.main.get_commands",
        MagicMock(
            return_value=[
                CommandInfo(
                    func=mock_command(non_default_command, "non_default_command"),
                    aliases=["command2"],
                    opt_groups=[non_default_command_params],
                    is_default=False,
                ),
            ]
        ),
    )

    main(["command2", "--param", "test"])
    non_default_command.assert_called_once_with(param="test")


def wrap_mock_command(mock: MagicMock, command: Callable) -> Callable:
    """Wrap the command with the mock."""

    @wraps(command)
    def mock_command(*args: Any, **kwargs: Any) -> Any:
        """Mock the command."""
        mock(*args, **kwargs)

    return mock_command


@pytest.mark.parametrize(
    "params, expected_call",
    [
        [
            ["check", "sample_model.tflite", "--target-profile", "ethos-u55-256"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.tflite",
                compatibility=False,
                performance=False,
                output=None,
                json=False,
                backend=None,
            ),
        ],
        [
            ["check", "sample_model.tflite", "--target-profile", "ethos-u55-128"],
            call(
                ctx=ANY,
                target_profile="ethos-u55-128",
                model="sample_model.tflite",
                compatibility=False,
                performance=False,
                output=None,
                json=False,
                backend=None,
            ),
        ],
        [
            [
                "check",
                "sample_model.h5",
                "--performance",
                "--compatibility",
                "--target-profile",
                "ethos-u55-256",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                output=None,
                json=False,
                compatibility=True,
                performance=True,
                backend=None,
            ),
        ],
        [
            [
                "check",
                "sample_model.h5",
                "--performance",
                "--target-profile",
                "ethos-u55-256",
                "--output",
                "result.json",
                "--json",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                performance=True,
                compatibility=False,
                output=Path("result.json"),
                json=True,
                backend=None,
            ),
        ],
        [
            [
                "check",
                "sample_model.h5",
                "--performance",
                "--target-profile",
                "ethos-u55-128",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-128",
                model="sample_model.h5",
                compatibility=False,
                performance=True,
                output=None,
                json=False,
                backend=None,
            ),
        ],
        [
            [
                "optimize",
                "sample_model.h5",
                "--target-profile",
                "ethos-u55-256",
                "--pruning",
                "--clustering",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                pruning=True,
                clustering=True,
                pruning_target=None,
                clustering_target=None,
                output=None,
                json=False,
                backend=None,
            ),
        ],
        [
            [
                "optimize",
                "sample_model.h5",
                "--target-profile",
                "ethos-u55-256",
                "--pruning",
                "--clustering",
                "--pruning-target",
                "0.5",
                "--clustering-target",
                "32",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                pruning=True,
                clustering=True,
                pruning_target=0.5,
                clustering_target=32,
                output=None,
                json=False,
                backend=None,
            ),
        ],
        [
            [
                "optimize",
                "sample_model.h5",
                "--target-profile",
                "ethos-u55-256",
                "--pruning",
                "--backend",
                "some_backend",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                pruning=True,
                clustering=False,
                pruning_target=None,
                clustering_target=None,
                output=None,
                json=False,
                backend=["some_backend"],
            ),
        ],
        [
            [
                "check",
                "sample_model.h5",
                "--compatibility",
                "--target-profile",
                "cortex-a",
            ],
            call(
                ctx=ANY,
                target_profile="cortex-a",
                model="sample_model.h5",
                compatibility=True,
                performance=False,
                output=None,
                json=False,
                backend=None,
            ),
        ],
    ],
)
def test_commands_execution(
    monkeypatch: pytest.MonkeyPatch, params: list[str], expected_call: Any
) -> None:
    """Test calling commands from the main function."""
    mock = MagicMock()

    monkeypatch.setattr(
        "mlia.cli.options.get_available_backends",
        MagicMock(return_value=["Vela", "some_backend"]),
    )

    for command in ["check", "optimize"]:
        monkeypatch.setattr(
            f"mlia.cli.main.{command}",
            wrap_mock_command(mock, getattr(mlia.cli.main, command)),
        )

    main(params)

    mock.assert_called_once_with(*expected_call.args, **expected_call.kwargs)


@pytest.mark.parametrize(
    "params, expected_call",
    [
        [
            ["list"],
            call(),
        ],
    ],
)
def test_commands_execution_backend_main(
    monkeypatch: pytest.MonkeyPatch,
    params: list[str],
    expected_call: Any,
) -> None:
    """Test calling commands from the backend_main function."""
    mock = MagicMock()

    monkeypatch.setattr(
        "mlia.cli.main.backend_list",
        wrap_mock_command(mock, getattr(mlia.cli.main, "backend_list")),
    )

    backend_main(params)

    mock.assert_called_once_with(*expected_call.args, **expected_call.kwargs)


@pytest.mark.parametrize(
    "debug, exc_mock, expected_output",
    [
        [
            True,
            MagicMock(side_effect=Exception("Error")),
            [
                "Execution finished with error: Error",
                "Please check the log files in the /tmp/mlia-",
                "/logs for more details",
            ],
        ],
        [
            False,
            MagicMock(side_effect=Exception("Error")),
            [
                "Execution finished with error: Error",
                "Please check the log files in the /tmp/mlia-",
                "/logs for more details, or enable debug mode (--debug)",
            ],
        ],
        [
            False,
            MagicMock(side_effect=KeyboardInterrupt()),
            ["Execution has been interrupted"],
        ],
        [
            False,
            MagicMock(
                side_effect=BackendUnavailableError(
                    "Backend sample is not available", "sample"
                )
            ),
            ["Error: Backend sample is not available."],
        ],
        [
            False,
            MagicMock(
                side_effect=BackendUnavailableError(
                    "Backend tosa-checker is not available", "tosa-checker"
                )
            ),
            [
                "Error: Backend tosa-checker is not available.",
                "Please use next command to install it: "
                'mlia-backend install "tosa-checker"',
            ],
        ],
        [
            False,
            MagicMock(side_effect=InternalError("Unknown error")),
            ["Internal error: Unknown error"],
        ],
    ],
)
def test_debug_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    debug: bool,
    exc_mock: MagicMock,
    expected_output: list[str],
) -> None:
    """Test flag --debug."""

    def command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for non default command."""
        parser.add_argument("--debug", action="store_true")

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
    if debug:
        params.append("--debug")

    exit_code = main(params)
    assert exit_code == 1

    stdout, _ = capsys.readouterr()
    for expected_message in expected_output:
        assert expected_message in stdout
