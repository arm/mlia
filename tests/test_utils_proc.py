# Copyright 2021, Arm Ltd.
"""Tests for the module utils/proc."""
# pylint: disable=no-self-use
import signal
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from mlia.utils.proc import CommandExecutor
from mlia.utils.proc import ExecutionFailed


class TestCommandExecutor:
    """Tests for class CommandExecutor."""

    def test_execute(self) -> None:
        """Test command execution."""
        executor = CommandExecutor()

        retcode, stdout, stderr = executor.execute(["echo", "hello world!"])
        assert retcode == 0
        assert stdout.decode().strip() == "hello world!"
        assert stderr.decode() == ""

    def test_submit(self) -> None:
        """Test command submittion."""
        executor = CommandExecutor()

        running_command = executor.submit(["sleep", "10"])
        assert running_command.is_alive() is True
        assert running_command.exit_code() is None

        running_command.kill()
        for _ in range(3):
            time.sleep(0.5)
            if not running_command.is_alive():
                break

        assert running_command.is_alive() is False
        assert running_command.exit_code() == -9

        with pytest.raises(ExecutionFailed):
            executor.execute(["sleep", "-1"])

    @pytest.mark.parametrize("wait", [True, False])
    def test_stop(self, wait: bool) -> None:
        """Test command termination."""
        executor = CommandExecutor()

        running_command = executor.submit(["sleep", "10"])
        running_command.stop(wait=wait)

        if wait:
            assert running_command.is_alive() is False

    def test_unable_to_stop(self, monkeypatch: Any) -> None:
        """Test when command could not be stopped."""
        running_command_mock = MagicMock()
        running_command_mock.poll.return_value = None

        monkeypatch.setattr(
            "mlia.utils.proc.subprocess.Popen",
            MagicMock(return_value=running_command_mock),
        )

        with pytest.raises(Exception, match="Unable to stop running command"):
            executor = CommandExecutor()
            running_command = executor.submit(["sleep", "10"])

            running_command.stop(num_of_attempts=1, interval=0.1)

        running_command_mock.send_signal.assert_called_once_with(signal.SIGINT)

    def test_stop_after_several_attempts(self, monkeypatch: Any) -> None:
        """Test when command could be stopped after several attempts."""
        running_command_mock = MagicMock()
        running_command_mock.poll.side_effect = [None, 0]

        monkeypatch.setattr(
            "mlia.utils.proc.subprocess.Popen",
            MagicMock(return_value=running_command_mock),
        )

        executor = CommandExecutor()
        running_command = executor.submit(["sleep", "10"])

        running_command.stop(num_of_attempts=1, interval=0.1)
        running_command_mock.send_signal.assert_called_once_with(signal.SIGINT)

    def test_send_signal(self) -> None:
        """Test sending signal."""
        executor = CommandExecutor()
        running_command = executor.submit(["sleep", "10"])
        running_command.send_signal(signal.SIGINT)

        # wait a bit for a signal processing
        time.sleep(1)

        assert running_command.is_alive() is False
        assert running_command.exit_code() == -2

    @pytest.mark.parametrize(
        "redirect_output, expected_output", [[True, "hello\n"], [False, ""]]
    )
    def test_wait(
        self, capsys: Any, redirect_output: bool, expected_output: str
    ) -> None:
        """Test wait completion functionality."""
        executor = CommandExecutor()

        running_command = executor.submit(["echo", "hello"])
        running_command.wait(redirect_output=redirect_output)

        out, _ = capsys.readouterr()
        assert out == expected_output
