# Copyright (C) 2021-2022, Arm Ltd.
"""Utils related to process management."""
import os
import signal
import subprocess
import time
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from contextlib import suppress
from pathlib import Path
from typing import Any
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple


class ExecutionFailed(Exception):
    """Execution failed exception."""

    def __init__(self, return_code: int, stdout: bytes, stderr: bytes) -> None:
        """Init ExecutionFailed exception instance."""
        super().__init__()

        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class OutputConsumer(ABC):
    """Base class for the output consumers."""

    @abstractmethod
    def feed(self, line: str) -> None:
        """Feed new line to the consumerr."""


class RunningCommand:
    """Running command."""

    def __init__(self, process: subprocess.Popen) -> None:
        """Init running command instance."""
        self.process = process
        self._output_consumers: Optional[List[OutputConsumer]] = None

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

    @property
    def output_consumers(self) -> Optional[List[OutputConsumer]]:
        """Property output_consumers."""
        return self._output_consumers

    @output_consumers.setter
    def output_consumers(self, output_consumers: List[OutputConsumer]) -> None:
        """Set output consumers."""
        self._output_consumers = output_consumers

    def consume_output(self) -> None:
        """Pass program's output to the consumers."""
        if self.process is None or self.output_consumers is None:
            return

        for line in self.stdout():
            for consumer in self.output_consumers:
                with suppress():
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
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise ExecutionFailed(result.returncode, result.stdout, result.stderr)

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
def working_directory(working_dir: Path) -> Generator[Path, None, None]:
    """Temporary change working directory."""
    current_working_dir = Path.cwd()
    os.chdir(working_dir)

    try:
        yield working_dir
    finally:
        os.chdir(current_working_dir)
