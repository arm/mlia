# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=attribute-defined-outside-init,no-self-use,not-callable
"""Pytests for testing aiet/utils/proc.py."""
from pathlib import Path
from typing import Any
from unittest import mock

import psutil
import pytest
from sh import ErrorReturnCode

from aiet.utils.proc import Command
from aiet.utils.proc import CommandFailedException
from aiet.utils.proc import CommandNotFound
from aiet.utils.proc import parse_command
from aiet.utils.proc import print_command_stdout
from aiet.utils.proc import run_and_wait
from aiet.utils.proc import save_process_info
from aiet.utils.proc import ShellCommand
from aiet.utils.proc import terminate_command
from aiet.utils.proc import terminate_external_process


class TestShellCommand:
    """Sample class for collecting tests."""

    def test_shellcommand_default_value(self) -> None:
        """Test the instantiation of the class ShellCommand with no parameter."""
        shell_command = ShellCommand()
        assert shell_command.base_log_path == "/tmp"

    @pytest.mark.parametrize(
        "base_log_path,expected", [("/test", "/test"), ("/asd", "/asd")]
    )
    def test_shellcommand_with_param(self, base_log_path: str, expected: str) -> None:
        """Test init ShellCommand with different parameters."""
        shell_command = ShellCommand(base_log_path)
        assert shell_command.base_log_path == expected

    def test_run_ls(self, monkeypatch: Any) -> None:
        """Test a simple ls command."""
        mock_command = mock.MagicMock()
        monkeypatch.setattr(Command, "bake", mock_command)

        mock_get_stdout_stderr_paths = mock.MagicMock()
        mock_get_stdout_stderr_paths.return_value = ("/tmp/std.out", "/tmp/std.err")
        monkeypatch.setattr(
            ShellCommand, "get_stdout_stderr_paths", mock_get_stdout_stderr_paths
        )

        shell_command = ShellCommand()
        shell_command.run("ls", "-l")
        assert mock_command.mock_calls[0] == mock.call(("-l",))
        assert mock_command.mock_calls[1] == mock.call()(
            _bg=True, _err="/tmp/std.err", _out="/tmp/std.out", _tee=True, _bg_exc=False
        )

    def test_run_command_not_found(self) -> None:
        """Test whe the command doesn't exist."""
        shell_command = ShellCommand()
        with pytest.raises(CommandNotFound):
            shell_command.run("lsl", "-l")

    def test_get_stdout_stderr_paths_valid_path(self) -> None:
        """Test the method to get files to store stdout and stderr."""
        valid_path = "/tmp"
        shell_command = ShellCommand(valid_path)
        out, err = shell_command.get_stdout_stderr_paths(valid_path, "cmd")
        assert out.exists() and out.is_file()
        assert err.exists() and err.is_file()
        assert "cmd" in out.name
        assert "cmd" in err.name

    def test_get_stdout_stderr_paths_not_invalid_path(self) -> None:
        """Test the method to get output files with an invalid path."""
        invalid_path = "/invalid/foo/bar"
        shell_command = ShellCommand(invalid_path)
        with pytest.raises(FileNotFoundError):
            shell_command.get_stdout_stderr_paths(invalid_path, "cmd")


@mock.patch("builtins.print")
def test_print_command_stdout_alive(mock_print: Any) -> None:
    """Test the print command stdout with an alive (running) process."""
    mock_command = mock.MagicMock()
    mock_command.is_alive.return_value = True
    mock_command.next.side_effect = ["test1", "test2", StopIteration]

    print_command_stdout(mock_command)

    mock_command.assert_has_calls(
        [mock.call.is_alive(), mock.call.next(), mock.call.next()]
    )
    mock_print.assert_has_calls(
        [mock.call("test1", end=""), mock.call("test2", end="")]
    )


@mock.patch("builtins.print")
def test_print_command_stdout_not_alive(mock_print: Any) -> None:
    """Test the print command stdout with a not alive (exited) process."""
    mock_command = mock.MagicMock()
    mock_command.is_alive.return_value = False
    mock_command.stdout = "test"

    print_command_stdout(mock_command)
    mock_command.assert_has_calls([mock.call.is_alive()])
    mock_print.assert_called_once_with("test")


def test_terminate_external_process_no_process(capsys: Any) -> None:
    """Test that non existed process could be terminated."""
    mock_command = mock.MagicMock()
    mock_command.terminate.side_effect = psutil.Error("Error!")

    terminate_external_process(mock_command)
    captured = capsys.readouterr()
    assert captured.out == "Unable to terminate process\n"


def test_terminate_external_process_case1() -> None:
    """Test when process terminated immediately."""
    mock_command = mock.MagicMock()
    mock_command.is_running.return_value = False

    terminate_external_process(mock_command)
    mock_command.terminate.assert_called_once()
    mock_command.is_running.assert_called_once()


def test_terminate_external_process_case2() -> None:
    """Test when process termination takes time."""
    mock_command = mock.MagicMock()
    mock_command.is_running.side_effect = [True, True, False]

    terminate_external_process(mock_command)
    mock_command.terminate.assert_called_once()
    assert mock_command.is_running.call_count == 3


def test_terminate_external_process_case3() -> None:
    """Test when process termination takes more time."""
    mock_command = mock.MagicMock()
    mock_command.is_running.side_effect = [True, True, True]

    terminate_external_process(
        mock_command, number_of_attempts=2, wait_period=0.1, wait_for_termination=0.1
    )
    assert mock_command.is_running.call_count == 3
    assert mock_command.terminate.call_count == 2


def test_terminate_external_process_case4() -> None:
    """Test when process termination takes more time."""
    mock_command = mock.MagicMock()
    mock_command.is_running.side_effect = [True, True, False]

    terminate_external_process(
        mock_command, number_of_attempts=2, wait_period=0.1, wait_for_termination=0.1
    )
    mock_command.terminate.assert_called_once()
    assert mock_command.is_running.call_count == 3
    assert mock_command.terminate.call_count == 1


def test_terminate_command_no_process() -> None:
    """Test command termination when process does not exist."""
    mock_command = mock.MagicMock()
    mock_command.process.signal_group.side_effect = ProcessLookupError()

    terminate_command(mock_command)
    mock_command.process.signal_group.assert_called_once()
    mock_command.is_alive.assert_not_called()


def test_terminate_command() -> None:
    """Test command termination."""
    mock_command = mock.MagicMock()
    mock_command.is_alive.return_value = False

    terminate_command(mock_command)
    mock_command.process.signal_group.assert_called_once()


def test_terminate_command_case1() -> None:
    """Test command termination when it takes time.."""
    mock_command = mock.MagicMock()
    mock_command.is_alive.side_effect = [True, True, False]

    terminate_command(mock_command, wait_period=0.1)
    mock_command.process.signal_group.assert_called_once()
    assert mock_command.is_alive.call_count == 3


def test_terminate_command_case2() -> None:
    """Test command termination when it takes much time.."""
    mock_command = mock.MagicMock()
    mock_command.is_alive.side_effect = [True, True, True]

    terminate_command(mock_command, number_of_attempts=3, wait_period=0.1)
    assert mock_command.is_alive.call_count == 3
    assert mock_command.process.signal_group.call_count == 2


class TestRunAndWait:
    """Test run_and_wait function."""

    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch: Any) -> None:
        """Init test method."""
        self.execute_command_mock = mock.MagicMock()
        monkeypatch.setattr(
            "aiet.utils.proc.execute_command", self.execute_command_mock
        )

        self.terminate_command_mock = mock.MagicMock()
        monkeypatch.setattr(
            "aiet.utils.proc.terminate_command", self.terminate_command_mock
        )

    def test_if_execute_command_raises_exception(self) -> None:
        """Test if execute_command fails."""
        self.execute_command_mock.side_effect = Exception("Error!")
        with pytest.raises(Exception, match="Error!"):
            run_and_wait("command", Path.cwd())

    def test_if_command_finishes_with_error(self) -> None:
        """Test if command finishes with error."""
        cmd_mock = mock.MagicMock()
        self.execute_command_mock.return_value = cmd_mock
        exit_code_mock = mock.PropertyMock(
            side_effect=ErrorReturnCode("cmd", bytearray(), bytearray())
        )
        type(cmd_mock).exit_code = exit_code_mock

        with pytest.raises(CommandFailedException):
            run_and_wait("command", Path.cwd())

    @pytest.mark.parametrize("terminate_on_error, call_count", ((False, 0), (True, 1)))
    def test_if_command_finishes_with_exception(
        self, terminate_on_error: bool, call_count: int
    ) -> None:
        """Test if command finishes with error."""
        cmd_mock = mock.MagicMock()
        self.execute_command_mock.return_value = cmd_mock
        exit_code_mock = mock.PropertyMock(side_effect=Exception("Error!"))
        type(cmd_mock).exit_code = exit_code_mock

        with pytest.raises(Exception, match="Error!"):
            run_and_wait("command", Path.cwd(), terminate_on_error=terminate_on_error)

        assert self.terminate_command_mock.call_count == call_count


def test_save_process_info_no_process(monkeypatch: Any, tmpdir: Any) -> None:
    """Test save_process_info function."""
    mock_process = mock.MagicMock()
    monkeypatch.setattr("psutil.Process", mock.MagicMock(return_value=mock_process))
    mock_process.children.side_effect = psutil.NoSuchProcess(555)

    pid_file_path = Path(tmpdir) / "test.pid"
    save_process_info(555, pid_file_path)
    assert not pid_file_path.exists()


def test_parse_command() -> None:
    """Test parse_command function."""
    assert parse_command("1.sh") == ["bash", "1.sh"]
    assert parse_command("1.sh", shell="sh") == ["sh", "1.sh"]
    assert parse_command("command") == ["command"]
    assert parse_command("command 123 --param=1") == ["command", "123", "--param=1"]
