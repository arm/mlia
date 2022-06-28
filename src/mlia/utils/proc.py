# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utils related to process management."""
import os
import signal
import subprocess
import time
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple


class OutputConsumer(ABC):
    """Base class for the output consumers."""

    @abstractmethod
    def feed(self, line: str) -> None:
        """Feed new line to the consumer."""


class RunningCommand:
    """Running command."""

    def __init__(self, process: subprocess.Popen) -> None:
        """Init running command instance."""
        self.process = process
        self.output_consumers: List[OutputConsumer] = []

    def is_alive(self) -> bool:
        """Return true if process is still alive."""
        return self.process.poll() is None

    def exit_code(self) -> Optional[int]:
        """Return process's return code."""
        return self.process.poll()

    def stdout(self) -> Iterable[str]:
        """Return std output of the process."""
        assert self.process.stdout is not None

        for line in self.process.stdout:
            yield line

    def kill(self) -> None:
        """Kill the process."""
        self.process.kill()

    def send_signal(self, signal_num: int) -> None:
        """Send signal to the process."""
        self.process.send_signal(signal_num)

    def consume_output(self) -> None:
        """Pass program's output to the consumers."""
        if self.process is None or not self.output_consumers:
            return

        for line in self.stdout():
            for consumer in self.output_consumers:
                consumer.feed(line)

    def stop(
        self, wait: bool = True, num_of_attempts: int = 5, interval: float = 0.5
    ) -> None:
        """Stop execution."""
        try:
            if not self.is_alive():
                return

            self.process.send_signal(signal.SIGINT)
            self.consume_output()

            if not wait:
                return

            for _ in range(num_of_attempts):
                time.sleep(interval)
                if not self.is_alive():
                    break
            else:
                raise Exception("Unable to stop running command")
        finally:
            self._close_fd()

    def _close_fd(self) -> None:
        """Close file descriptors."""

        def close(file_descriptor: Any) -> None:
            """Check and close file."""
            if file_descriptor is not None and hasattr(file_descriptor, "close"):
                file_descriptor.close()

        close(self.process.stdout)
        close(self.process.stderr)

    def wait(self, redirect_output: bool = False) -> None:
        """Redirect process output to stdout and wait for completion."""
        if redirect_output:
            for line in self.stdout():
                print(line, end="")

        self.process.wait()


class CommandExecutor:
    """Command executor."""

    @staticmethod
    def execute(command: List[str]) -> Tuple[int, bytes, bytes]:
        """Execute the command."""
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        return (result.returncode, result.stdout, result.stderr)

    @staticmethod
    def submit(command: List[str]) -> RunningCommand:
        """Submit command for the execution."""
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # redirect command stderr to stdout
            universal_newlines=True,
            bufsize=1,
        )

        return RunningCommand(process)


@contextmanager
def working_directory(
    working_dir: Path, create_dir: bool = False
) -> Generator[Path, None, None]:
    """Temporary change working directory."""
    current_working_dir = Path.cwd()

    if create_dir:
        working_dir.mkdir()

    os.chdir(working_dir)

    try:
        yield working_dir
    finally:
        os.chdir(current_working_dir)
