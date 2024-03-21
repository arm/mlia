# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for process management."""
from __future__ import annotations

import logging
import subprocess  # nosec
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Generator


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Command:
    """Command information."""

    cmd: list[str]
    cwd: Path = Path.cwd()
    env: dict[str, str] | None = None


def command_output(command: Command) -> Generator[str, None, None]:
    """Get command output."""
    logger.debug("Running command: %s", command)

    with subprocess.Popen(  # nosec
        command.cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        cwd=command.cwd,
        env=command.env,
    ) as process:
        yield from process.stdout or []

    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command.cmd)


OutputConsumer = Callable[[str], None]


class OutputLogger:
    """Log process output to the given logger with the given level."""

    def __init__(self, logger_: logging.Logger, level: int = logging.DEBUG) -> None:
        """Create log function with the appropriate log level set."""
        self.log_fn = partial(logger_.log, level)

    def __call__(self, line: str) -> None:
        """Redirect output to the logger."""
        self.log_fn(line)


def process_command_output(
    command: Command,
    consumers: list[OutputConsumer],
) -> None:
    """Execute command and process output."""
    for line in command_output(command):
        for consumer in consumers:
            consumer(line)
