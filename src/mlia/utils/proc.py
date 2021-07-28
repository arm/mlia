# Copyright 2021, Arm Ltd.
"""Utils related to process management."""
import signal
import subprocess
import time
from abc import ABC
from abc import abstractmethod
from contextlib import suppress
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple


class ExecutionFailed(Exception):
    """Execution failed exception."""

    def __init__(self, return_code: int, stdout: bytes, stderr: bytes) -> None:
        """Init ExecutionFailed exception instance."""
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

    def terminate(self) -> None:
        """Terminate the process."""
        self.process.terminate()

    def send_signal(self, signal: int) -> None:
        """Send signal to the process."""
        self.process.send_signal(signal)

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

        def close(f: Any) -> None:
            """Check and close file."""
            if f is not None and hasattr(f, "close"):
                f.close()

        close(self.process.stdout)
        close(self.process.stderr)

    def wait(self, redirect_output: bool = False) -> None:
        """Redirect process output to stdout and wait for completition."""
        if redirect_output:
            for line in self.stdout():
                print(line, end="")

        self.process.wait()


class CommandExecutor:
    """Command executor."""

    def execute(self, command: List[str]) -> Tuple[int, bytes, bytes]:
        """Execute the command."""
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise ExecutionFailed(result.returncode, result.stdout, result.stderr)

        return (result.returncode, result.stdout, result.stderr)

    def submit(self, command: List[str]) -> RunningCommand:
        """Submit command for the execution."""
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # redirect command stderr to stdout
            universal_newlines=True,
            bufsize=1,
        )

        return RunningCommand(process)
