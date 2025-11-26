# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
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

import mlia.cli.commands
import mlia.cli.main as mlia_cli_main
from mlia.backend.errors import BackendUnavailableError
from mlia.cli.main import backend_main
from mlia.cli.main import CommandInfo
from mlia.cli.main import get_possible_command_names
from mlia.cli.main import main
from mlia.cli.main import target_main
from mlia.cli.options import add_output_directory
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
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


def test_command_info() -> None:
    """Test properties of CommandInfo object."""

    def test_command() -> None:
        """Test command."""

    command_info = CommandInfo(test_command, ["test"], [])
    assert command_info.command_name == "test_command"
    assert command_info.command_name_and_aliases == ["test_command", "test"]
    assert command_info.command_help == "Test command"


def test_get_possible_command_names() -> None:
    """Test get_possible_command_names returns all command names and aliases."""

    def command_one() -> None:
        """First command."""

    def command_two() -> None:
        """Second command."""

    def third_cmd() -> None:
        """Third command."""

    commands = [
        CommandInfo(command_one, ["c1", "cmd1"], []),
        CommandInfo(command_two, [], []),
        CommandInfo(third_cmd, ["t3"], []),
    ]

    result = get_possible_command_names(commands)

    assert result == [
        "command_one",
        "c1",
        "cmd1",
        "command_two",
        "third_cmd",
        "t3",
    ]


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
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                performance=True,
                compatibility=False,
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
                optimization_profile="optimization",
                backend=None,
                rewrite=False,
                rewrite_target=None,
                rewrite_start=None,
                rewrite_end=None,
                dataset=None,
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
                backend=None,
                optimization_profile="optimization",
                rewrite=False,
                rewrite_target=None,
                rewrite_start=None,
                rewrite_end=None,
                dataset=None,
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
                optimization_profile="optimization",
                backend=["some_backend"],
                rewrite=False,
                rewrite_target=None,
                rewrite_start=None,
                rewrite_end=None,
                dataset=None,
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
                "--optimization-profile",
                "optimization",
            ],
            call(
                ctx=ANY,
                target_profile="ethos-u55-256",
                model="sample_model.h5",
                pruning=True,
                clustering=False,
                pruning_target=None,
                clustering_target=None,
                backend=["some_backend"],
                optimization_profile="optimization",
                rewrite=False,
                rewrite_target=None,
                rewrite_start=None,
                rewrite_end=None,
                dataset=None,
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
        MagicMock(return_value=["vela", "some_backend"]),
    )

    for command in ["check", "optimize"]:
        monkeypatch.setattr(
            f"mlia.cli.main.{command}",
            wrap_mock_command(mock, getattr(mlia_cli_main, command)),
        )

    main(params)

    mock.assert_called_once_with(*expected_call.args, **expected_call.kwargs)


def test_passing_output_directory_parameter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test passing parameter --output-dir."""
    passed_context: ExecutionContext | None = None

    def sample_command(ctx: ExecutionContext) -> None:
        """Sample command."""
        nonlocal passed_context
        passed_context = ctx

    monkeypatch.setattr(
        "mlia.cli.main.get_commands",
        lambda: [CommandInfo(sample_command, [], [add_output_directory])],
    )

    output_dir = tmp_path / "output"
    main(["sample_command", "--output-dir", output_dir.as_posix()])

    assert isinstance(passed_context, ExecutionContext)
    assert passed_context.output_dir == output_dir / "mlia-output"


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
        wrap_mock_command(mock, getattr(mlia_cli_main, "backend_list")),
    )

    backend_main(params)

    mock.assert_called_once_with(*expected_call.args, **expected_call.kwargs)


@pytest.mark.parametrize(
    "params, expected_call",
    [
        (["list"], call()),
    ],
)
def test_commands_execution_target_main(
    monkeypatch: pytest.MonkeyPatch,
    params: list[str],
    expected_call: Any,
) -> None:
    """Test calling commands from the target_main function."""
    mock = MagicMock()

    monkeypatch.setattr(
        "mlia.cli.main.target_list",
        wrap_mock_command(mock, getattr(mlia.cli.commands, "target_list")),
    )

    target_main(params)

    mock.assert_called_once_with(*expected_call.args, **expected_call.kwargs)


# mypy: disable-error-code=misc
@pytest.mark.parametrize(
    "debug, exc_mock, expected_output",
    [
        [
            True,
            MagicMock(side_effect=Exception("Error")),
            [
                "Execution finished with error: Error",
                "Please check the log files in the",
                "/logs for more details",
            ],
        ],
        [
            False,
            MagicMock(side_effect=Exception("Error")),
            [
                "Execution finished with error: Error",
                "Please check the log files in the",
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
            MagicMock(
                side_effect=BackendUnavailableError(
                    "Backend vela is not available", "vela"
                )
            ),
            [
                "Error: Backend vela is not available.",
                'Please use next command to install it: mlia-backend install "vela"',
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


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("Init failed"),
        ConfigurationError("Config broken"),
    ],
    ids=["runtime_error", "configuration_error"],
)
def test_setup_context_exception_handling(
    exception: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test that exceptions during ExecutionContext init are caught and reported.

    The broad except in setup_context should print the error to stderr and
    exit with code 1 when ExecutionContext raises an exception.
    """

    # Provide a minimal command definition so parsing succeeds
    def sample_command(_ctx: ExecutionContext) -> None:
        """Sample command (unused because context creation fails)."""

    monkeypatch.setattr(
        mlia_cli_main, "get_commands", lambda: [CommandInfo(sample_command, [], [])]
    )

    # Force ExecutionContext to raise RuntimeError during setup_context
    monkeypatch.setattr(
        mlia_cli_main,
        "ExecutionContext",
        MagicMock(side_effect=exception),
    )

    with pytest.raises(SystemExit) as ex:
        main(["sample_command"])

    assert ex.value.code == 1
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert str(exception) in stderr


def test_run_command_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """
    Test that ConfigurationError during command execution is caught in run_command.
    """

    def sample_command(ctx: ExecutionContext) -> None:
        """Sample command that raises ConfigurationError."""
        raise ConfigurationError("Configuration is invalid")

    monkeypatch.setattr(
        mlia_cli_main, "get_commands", lambda: [CommandInfo(sample_command, [], [])]
    )

    exit_code = main(["sample_command"])

    assert exit_code == 1
    stdout, _ = capsys.readouterr()
    assert "Configuration is invalid" in stdout
