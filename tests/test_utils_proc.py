# Copyright 2021, Arm Ltd.
"""Tests for the module utils/proc."""
# pylint: disable=no-self-use
import time
from typing import Any

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

    def test_stop(self) -> None:
        """Test command termination."""
        executor = CommandExecutor()

        running_command = executor.submit(["sleep", "10"])
        running_command.stop()

        assert running_command.is_alive() is False

    def test_wait(self, capsys: Any) -> None:
        """Test wait completion functionality."""
        executor = CommandExecutor()

        running_command = executor.submit(["echo", "hello"])
        running_command.wait(redirect_output=True)

        out, _ = capsys.readouterr()
        assert out == "hello\n"
