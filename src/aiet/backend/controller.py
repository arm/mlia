# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Controller backend module."""
import time
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import psutil
import sh

from aiet.backend.common import ConfigurationException
from aiet.utils.fs import read_file_as_string
from aiet.utils.proc import execute_command
from aiet.utils.proc import get_stdout_stderr_paths
from aiet.utils.proc import read_process_info
from aiet.utils.proc import save_process_info
from aiet.utils.proc import terminate_command
from aiet.utils.proc import terminate_external_process


class SystemController:
    """System controller class."""

    def __init__(self) -> None:
        """Create new instance of service controller."""
        self.cmd: Optional[sh.RunningCommand] = None
        self.out_path: Optional[Path] = None
        self.err_path: Optional[Path] = None

    def before_start(self) -> None:
        """Run actions before system start."""

    def after_start(self) -> None:
        """Run actions after system start."""

    def start(self, commands: List[str], cwd: Path) -> None:
        """Start system."""
        if not isinstance(cwd, Path) or not cwd.is_dir():
            raise ConfigurationException("Wrong working directory {}".format(cwd))

        if len(commands) != 1:
            raise ConfigurationException("System should have only one command to run")

        startup_command = commands[0]
        if not startup_command:
            raise ConfigurationException("No startup command provided")

        self.before_start()

        self.out_path, self.err_path = get_stdout_stderr_paths(startup_command)

        self.cmd = execute_command(
            startup_command,
            cwd,
            bg=True,
            out=str(self.out_path),
            err=str(self.err_path),
        )

        self.after_start()

    def stop(
        self, wait: bool = False, wait_period: float = 0.5, number_of_attempts: int = 20
    ) -> None:
        """Stop system."""
        if self.cmd is not None and self.is_running():
            terminate_command(self.cmd, wait, wait_period, number_of_attempts)

    def is_running(self) -> bool:
        """Check if underlying process is running."""
        return self.cmd is not None and self.cmd.is_alive()

    def get_output(self) -> Tuple[str, str]:
        """Return application output."""
        if self.cmd is None or self.out_path is None or self.err_path is None:
            return ("", "")

        return (read_file_as_string(self.out_path), read_file_as_string(self.err_path))


class SystemControllerSingleInstance(SystemController):
    """System controller with support of system's single instance."""

    def __init__(self, pid_file_path: Optional[Path] = None) -> None:
        """Create new instance of the service controller."""
        super().__init__()
        self.pid_file_path = pid_file_path

    def before_start(self) -> None:
        """Run actions before system start."""
        self._check_if_previous_instance_is_running()

    def after_start(self) -> None:
        """Run actions after system start."""
        self._save_process_info()

    def _check_if_previous_instance_is_running(self) -> None:
        """Check if another instance of the system is running."""
        process_info = read_process_info(self._pid_file())

        for item in process_info:
            try:
                process = psutil.Process(item.pid)
                same_process = (
                    process.name() == item.name
                    and process.exe() == item.executable
                    and process.cwd() == item.cwd
                )
                if same_process:
                    print(
                        "Stopping previous instance of the system [{}]".format(item.pid)
                    )
                    terminate_external_process(process)
            except psutil.NoSuchProcess:
                pass

    def _save_process_info(self, wait_period: float = 2) -> None:
        """Save information about system's processes."""
        if self.cmd is None or not self.is_running():
            return

        # give some time for the system to start
        time.sleep(wait_period)

        save_process_info(self.cmd.process.pid, self._pid_file())

    def _pid_file(self) -> Path:
        """Return path to file which is used for saving process info."""
        if not self.pid_file_path:
            raise Exception("No pid file path presented")

        return self.pid_file_path
