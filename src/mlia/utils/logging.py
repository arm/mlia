# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Logging utility functions."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from contextlib import ExitStack
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import TextIO


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
    stderr_level: int = logging.ERROR,
) -> Generator[None, None, None]:
    """Redirect standard output to the logger."""
    stdout_to_log = LoggerWriter(logger, stdout_level)
    stderr_to_log = LoggerWriter(logger, stderr_level)

    with ExitStack() as exit_stack:
        exit_stack.enter_context(redirect_stdout(stdout_to_log))  # type: ignore
        exit_stack.enter_context(redirect_stderr(stderr_to_log))  # type: ignore

        yield


@contextmanager
def redirect_raw(
    logger: logging.Logger, output: TextIO, log_level: int
) -> Generator[None, None, None]:
    """Redirect output using file descriptors."""
    with tempfile.TemporaryFile(mode="r+") as tmp:
        old_output_fd: int | None = None
        try:
            output_fd = output.fileno()
            old_output_fd = os.dup(output_fd)
            os.dup2(tmp.fileno(), output_fd)

            yield
        finally:
            if old_output_fd is not None:
                os.dup2(old_output_fd, output_fd)
                os.close(old_output_fd)

            tmp.seek(0)
            for line in tmp.readlines():
                logger.log(log_level, line.rstrip())


@contextmanager
def redirect_raw_output(
    logger: logging.Logger,
    stdout_level: int | None = logging.INFO,
    stderr_level: int | None = logging.ERROR,
) -> Generator[None, None, None]:
    """Redirect output on the process level."""
    with ExitStack() as exit_stack:
        for level, output in [
            (stdout_level, sys.stdout),
            (stderr_level, sys.stderr),
        ]:
            if level is not None:
                exit_stack.enter_context(redirect_raw(logger, output, level))

        yield


class LogFilter(logging.Filter):
    """Configurable log filter."""

    def __init__(self, log_record_filter: Callable[[logging.LogRecord], bool]) -> None:
        """Init log filter instance."""
        super().__init__()
        self.log_record_filter = log_record_filter

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log messages."""
        return self.log_record_filter(record)

    @classmethod
    def equals(cls, log_level: int) -> LogFilter:
        """Return log filter that filters messages by log level."""

        def filter_by_level(log_record: logging.LogRecord) -> bool:
            return log_record.levelno == log_level

        return cls(filter_by_level)

    @classmethod
    def skip(cls, log_level: int) -> LogFilter:
        """Return log filter that skips messages with particular level."""

        def skip_by_level(log_record: logging.LogRecord) -> bool:
            return log_record.levelno != log_level

        return cls(skip_by_level)


def create_log_handler(
    *,
    file_path: Path | None = None,
    stream: Any | None = None,
    log_level: int | None = None,
    log_format: str | None = None,
    log_filter: logging.Filter | None = None,
    delay: bool = True,
) -> logging.Handler:
    """Create logger handler."""
    handler: logging.Handler | None = None

    if file_path is not None:
        handler = logging.FileHandler(file_path, delay=delay)
    elif stream is not None:
        handler = logging.StreamHandler(stream)

    if handler is None:
        raise Exception("Unable to create logging handler")

    if log_level:
        handler.setLevel(log_level)

    if log_format:
        handler.setFormatter(logging.Formatter(log_format))

    if log_filter:
        handler.addFilter(log_filter)

    return handler


def attach_handlers(
    handlers: Iterable[logging.Handler], loggers: Iterable[logging.Logger]
) -> None:
    """Attach handlers to the loggers."""
    for handler in handlers:
        for logger in loggers:
            logger.addHandler(handler)


@contextmanager
def log_action(action: str) -> Generator[None, None, None]:
    """Log action."""
    logger = logging.getLogger(__name__)

    logger.info(action)
    yield
    logger.info("Done\n")
