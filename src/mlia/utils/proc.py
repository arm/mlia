# SPDX-FileCopyrightText: Copyright 2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for process management."""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable

from mlia.utils.boundaries.process import Command, command_output

logger = logging.getLogger(__name__)


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
