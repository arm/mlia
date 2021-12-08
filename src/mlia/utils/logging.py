# Copyright 2021, Arm Ltd.
"""Logging utility functions."""
import logging
from contextlib import contextmanager
from contextlib import ExitStack
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from typing import Generator


class LoggerWriter:
    """Redirect printed messages to the logger."""

    def __init__(self, logger: logging.Logger, level: int):
        """Init logger writer."""
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        """Write message."""
        if message.strip() != "":
            self.logger.log(self.level, message)

    def flush(self) -> None:
        """Flush buffers."""


@contextmanager
def redirect_output(
    logger: logging.Logger,
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.INFO,
) -> Generator[None, None, None]:
    """Redirect standard output to the logger."""
    stdout_to_log = LoggerWriter(logger, stdout_level)
    stderr_to_log = LoggerWriter(logger, stderr_level)

    with ExitStack() as exit_stack:
        exit_stack.enter_context(redirect_stdout(stdout_to_log))  # type: ignore
        exit_stack.enter_context(redirect_stderr(stderr_to_log))  # type: ignore

        yield
