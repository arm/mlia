# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI main entry point."""

from __future__ import annotations

import os
import sys

import typer
from rich.console import Console

import mlia.cli.commands as cli_commands
from mlia.cli.commands import (
    backend_app,
    mlia_app,
    target_app,
)
from mlia.plugins.plugins import load_cli_plugins as _load_cli_plugins

DEPRECATED_BACKEND_ENTRY_POINT = (
    "Warning: 'mlia-backend' is deprecated. Use 'mlia backend' instead."
)
DEPRECATED_TARGET_ENTRY_POINT = (
    "Warning: 'mlia-target' is deprecated. Use 'mlia target' instead."
)


def load_cli_plugins(*args: object) -> None:
    """Keep the legacy plugin loader import path available."""
    _load_cli_plugins(*args)


def _no_color_enabled() -> bool:
    """Return whether CLI colors should be disabled."""
    no_color = os.getenv("NO_COLOR")
    return (no_color is not None and no_color != "") or not sys.stdout.isatty()


def _configure_cli_colors() -> bool:
    """Configure shared CLI color handling and return whether color is enabled."""
    no_color = _no_color_enabled()
    cli_commands.console = Console(no_color=no_color)
    return not no_color


def main() -> None:
    """Entry point of the main application."""
    color = _configure_cli_colors()
    mlia_app(color=color)


def backend_main() -> None:
    """Entry point of the backend application."""
    color = _configure_cli_colors()
    typer.secho(
        DEPRECATED_BACKEND_ENTRY_POINT,
        fg=typer.colors.YELLOW,
        color=color,
        err=True,
    )
    backend_app(color=color)


def target_main() -> None:
    """Entry point of the target application."""
    color = _configure_cli_colors()
    typer.secho(
        DEPRECATED_TARGET_ENTRY_POINT,
        fg=typer.colors.YELLOW,
        color=color,
        err=True,
    )
    target_app(color=color)


if __name__ == "__main__":  # pragma: no cover
    main()
