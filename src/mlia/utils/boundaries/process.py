# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified process boundary gateways."""

from __future__ import annotations

import logging
import subprocess  # nosec
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from mlia.utils.logging import (
    log_boundary_action,
    log_progress_action,
    progress_heartbeat,
)


@dataclass(frozen=True)
class Command:
    """Command information."""

    cmd: list[str]
    cwd: Path = Path.cwd()
    env: dict[str, str] | None = None


def command_output(command: Command) -> Generator[str, None, None]:
    """Run an approved command and yield its output."""
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


def run_pip_with_notice(
    logger: logging.Logger, subcommand: str, params: list[str]
) -> tuple[str, int]:
    """Announce and run pip without logging its potentially sensitive arguments."""
    assert sys.executable, "Unable to launch pip command"
    log_boundary_action(
        logger,
        "Running pip %s using Python interpreter '%s'.",
        subcommand,
        sys.executable,
    )
    if subcommand == "install":
        log_progress_action(logger, "Installing Python packages...")

    try:
        label = (
            "Python package installation"
            if subcommand == "install"
            else "Python package removal"
        )
        with progress_heartbeat(logger, label):
            output = subprocess.check_output(  # nosec
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    subcommand,
                    *params,
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )

        if subcommand == "install":
            log_progress_action(logger, "Python packages installed.")

        return output, 0
    except subprocess.CalledProcessError as err:
        return err.output, err.returncode
