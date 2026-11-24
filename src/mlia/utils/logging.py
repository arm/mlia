# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Logging utility functions."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, TextIO

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from mlia.utils.boundaries.logging import create_file_handler
from mlia.utils.boundaries.temporary import temporary_text_stream
from mlia.utils.console import remove_ascii_codes

_INTERACTIVE_PROGRESS: ContextVar[bool] = ContextVar(
    "mlia_interactive_progress", default=False
)
_PROGRESS_CONSOLE: ContextVar[Console | None] = ContextVar(
    "mlia_progress_console", default=None
)
_LIVE_TASK: ContextVar[tuple[Progress, TaskID] | None] = ContextVar(
    "mlia_live_progress_task", default=None
)
_CURRENT_PROGRESS_LABEL: ContextVar[str | None] = ContextVar(
    "mlia_current_progress_label", default=None
)
_ACTIVE_PROGRESS: ContextVar[bool] = ContextVar("mlia_active_progress", default=False)
_TEXT_PROGRESS_LABELS: ContextVar[list[str] | None] = ContextVar(
    "mlia_text_progress_labels", default=None
)


def configure_interactive_progress(
    enabled: bool, console: Console | None = None
) -> None:
    """Select an animated indicator for human-facing terminal output."""
    _INTERACTIVE_PROGRESS.set(enabled)
    _PROGRESS_CONSOLE.set(console)


class ProgressStreamHandler(logging.StreamHandler):
    """Write stage messages through the configured console when present."""

    def __init__(self, stream: TextIO, console: Console | None = None) -> None:
        """Keep the configured console available to heartbeat worker threads."""
        super().__init__(stream)
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the spinner and log line from writing over one another."""
        console = self.console
        if console is None:
            super().emit(record)
            return
        try:
            console.print(self.format(record), markup=False, highlight=False)
        except Exception:  # pragma: no cover - logging handlers must not raise
            self.handleError(record)


def log_boundary_action(logger: logging.Logger, message: str, *args: object) -> None:
    """Log an externally visible runtime action."""
    logger.info(message, *args, extra={"boundary_action": True})


def log_progress_action(logger: logging.Logger, message: str, *args: object) -> None:
    """Log a short check stage for the ordinary CLI and the log file."""
    logger.info(message, *args, extra={"progress_action": True})


@contextmanager
def progress_heartbeat(
    logger: logging.Logger, label: str, interval: float = 10.0
) -> Generator[None, None, None]:
    """Show a transient spinner or timed text while a step is quiet."""
    if _ACTIVE_PROGRESS.get():
        live_task = _LIVE_TASK.get()
        if live_task is None:
            labels = _TEXT_PROGRESS_LABELS.get()
            if labels is None:
                yield
                return
            labels.append(label)
            try:
                yield
            finally:
                labels.pop()
            return
        progress, task_id = live_task
        previous_label = _CURRENT_PROGRESS_LABEL.get()
        label_token = _CURRENT_PROGRESS_LABEL.set(label)
        try:
            progress.update(task_id, description=label, refresh=True)
            yield
        finally:
            _CURRENT_PROGRESS_LABEL.reset(label_token)
            if previous_label is not None:
                progress.update(task_id, description=previous_label, refresh=True)
        return
    token = _ACTIVE_PROGRESS.set(True)
    try:
        if _INTERACTIVE_PROGRESS.get():
            console = _PROGRESS_CONSOLE.get()
            if console is not None and console.is_terminal:
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("{task.description}", markup=False),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True,
                    redirect_stdout=True,
                    redirect_stderr=True,
                ) as progress:
                    task_id = progress.add_task(label, total=None)
                    task_token = _LIVE_TASK.set((progress, task_id))
                    label_token = _CURRENT_PROGRESS_LABEL.set(label)
                    try:
                        yield
                    finally:
                        _CURRENT_PROGRESS_LABEL.reset(label_token)
                        _LIVE_TASK.reset(task_token)
                return

        stopped = threading.Event()
        started = time.monotonic()
        labels = [label]
        labels_token = _TEXT_PROGRESS_LABELS.set(labels)

        def report() -> None:
            while not stopped.wait(interval):
                log_progress_action(
                    logger,
                    "%s still in progress (%ds elapsed).",
                    labels[-1],
                    int(time.monotonic() - started),
                )

        thread = threading.Thread(target=report, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join()
            _TEXT_PROGRESS_LABELS.reset(labels_token)
    finally:
        _ACTIVE_PROGRESS.reset(token)


class LoggerWriter:
    """Redirect printed messages to the logger."""

    def __init__(self, logger: logging.Logger, level: int):
        """Init logger writer."""
        self.logger = logger
        self.level = level
        self._write_state = threading.local()

    def write(self, message: str) -> None:
        """Write message."""
        if not message.strip():
            return

        # Avoid recursive logging when a handler stream points back to this
        # writer (e.g. nested redirections in tool runtimes).
        if getattr(self._write_state, "active", False):
            return
        self._write_state.active = True
        try:
            self.logger.log(self.level, message)
        finally:
            self._write_state.active = False

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
def process_raw_output(
    consumer: Callable[[str], None], output: TextIO
) -> Generator[None, None, None]:
    """Process output on file descriptor level."""
    with temporary_text_stream() as tmp:
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
                consumer(line)


@contextmanager
def redirect_raw(
    logger: logging.Logger, output: TextIO, log_level: int
) -> Generator[None, None, None]:
    """Redirect output using file descriptors."""

    def consumer(line: str) -> None:
        """Redirect output to the logger."""
        logger.log(log_level, line.rstrip())

    with process_raw_output(consumer, output):
        yield


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


@contextmanager
def capture_raw_output(output: TextIO) -> Generator[list[str], None, None]:
    """Capture output as list of strings."""
    result: list[str] = []

    def consumer(line: str) -> None:
        """Save output for later processing."""
        result.append(line)

    with process_raw_output(consumer, output):
        yield result


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


class NoASCIIFormatter(logging.Formatter):
    """Custom Formatter for logging into file."""

    def format(self, record: logging.LogRecord) -> str:
        """Overwrite format method to remove ascii codes from record."""
        result = super().format(record)
        return remove_ascii_codes(result)


def create_log_handler(
    *,
    file_path: Path | None = None,
    stream: Any | None = None,
    log_level: int | None = None,
    log_format: str | logging.Formatter | None = None,
    log_filter: logging.Filter | None = None,
    delay: bool = True,
    progress_safe: bool = False,
    console: Console | None = None,
) -> logging.Handler:
    """Create logger handler."""
    handler: logging.Handler | None = None

    if file_path is not None:
        handler = create_file_handler(file_path, delay=delay)
    elif stream is not None:
        handler = (
            ProgressStreamHandler(stream, console)
            if progress_safe
            else logging.StreamHandler(stream)
        )

    if handler is None:
        raise RuntimeError("Unable to create logging handler.")

    if log_level:
        handler.setLevel(log_level)

    if log_format:
        if isinstance(log_format, str):
            log_format = logging.Formatter(log_format)
        handler.setFormatter(log_format)

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
    """Log an action and identify the work completed on success."""
    logger = logging.getLogger(__name__)

    logger.info(action)
    yield
    logger.info("Completed: %s", action.rstrip(". "))
