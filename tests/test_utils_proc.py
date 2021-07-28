# Copyright 2021, Arm Ltd.
"""Tests for the module utils/proc."""
import time

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
