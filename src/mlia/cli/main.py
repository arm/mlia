# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI main entry point."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from mlia.cli.commands import backend_app, color_enabled, mlia_app, target_app
from mlia.plugins.plugins import load_cli_plugins as _load_cli_plugins
from mlia.utils.registry import Registry

DEPRECATED_BACKEND_ENTRY_POINT = (
    "Warning: 'mlia-backend' is deprecated. Use 'mlia backend' instead."
)
DEPRECATED_TARGET_ENTRY_POINT = (
    "Warning: 'mlia-target' is deprecated. Use 'mlia target' instead."
)


def load_cli_plugins(registry: Registry[object]) -> None:
    """Keep the legacy plugin loader import path available."""
    _load_cli_plugins(registry)


def main() -> None:
    """Entry point of the main application."""
    load_dotenv()
    color = color_enabled()
    mlia_app(color=color)


def backend_main() -> None:
    """Entry point of the backend application."""
    load_dotenv()
    color = color_enabled()
    typer.secho(
        DEPRECATED_BACKEND_ENTRY_POINT,
        fg=typer.colors.YELLOW,
        color=color,
        err=True,
    )
    backend_app(color=color)


def target_main() -> None:
    """Entry point of the target application."""
    load_dotenv()
    color = color_enabled()
    typer.secho(
        DEPRECATED_TARGET_ENTRY_POINT,
        fg=typer.colors.YELLOW,
        color=color,
        err=True,
    )
    target_app(color=color)


if __name__ == "__main__":  # pragma: no cover
    main()
