# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module utils/proc."""
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.utils.proc import CommandExecutor
from mlia.utils.proc import working_directory


class TestCommandExecutor:
    """Tests for class CommandExecutor."""

    @staticmethod
    def test_execute() -> None:
        """Test command execution."""
        executor = CommandExecutor()

        retcode, stdout, stderr = executor.execute(["echo", "hello world!"])
        assert retcode == 0
        assert stdout.decode().strip() == "hello world!"
        assert stderr.decode() == ""

    @staticmethod
    def test_submit() -> None:
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

        with pytest.raises(subprocess.CalledProcessError):
            executor.execute(["sleep", "-1"])

    @staticmethod
    @pytest.mark.parametrize("wait", [True, False])
    def test_stop(wait: bool) -> None:
        """Test command termination."""
        executor = CommandExecutor()

        running_command = executor.submit(["sleep", "10"])
        running_command.stop(wait=wait)

        if wait:
            assert running_command.is_alive() is False

    @staticmethod
    def test_unable_to_stop(monkeypatch: pytest.MonkeyPatch) -> None:
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

    @staticmethod
    def test_stop_after_several_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
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

    @staticmethod
    def test_send_signal() -> None:
        """Test sending signal."""
        executor = CommandExecutor()
        running_command = executor.submit(["sleep", "10"])
        running_command.send_signal(signal.SIGINT)

        # wait a bit for a signal processing
        time.sleep(1)

        assert running_command.is_alive() is False
        assert running_command.exit_code() == -2

    @staticmethod
    @pytest.mark.parametrize(
        "redirect_output, expected_output", [[True, "hello\n"], [False, ""]]
    )
    def test_wait(
        capsys: pytest.CaptureFixture, redirect_output: bool, expected_output: str
    ) -> None:
        """Test wait completion functionality."""
        executor = CommandExecutor()

        running_command = executor.submit(["echo", "hello"])
        running_command.wait(redirect_output=redirect_output)

        out, _ = capsys.readouterr()
        assert out == expected_output


@pytest.mark.parametrize(
    "should_exist, create_dir",
    [
        [True, False],
        [False, True],
    ],
)
def test_working_directory_context_manager(
    tmp_path: Path, should_exist: bool, create_dir: bool
) -> None:
    """Test working_directory context manager."""
    prev_wd = Path.cwd()

    working_dir = tmp_path / "work_dir"
    if should_exist:
        working_dir.mkdir()

    with working_directory(working_dir, create_dir=create_dir) as current_working_dir:
        assert current_working_dir.is_dir()
        assert Path.cwd() == current_working_dir

    assert Path.cwd() == prev_wd
