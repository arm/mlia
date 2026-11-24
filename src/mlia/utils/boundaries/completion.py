# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified gateway for persistent shell-completion setup."""

from __future__ import annotations

import logging
from typing import Any

import click
from typer.completion import install_callback

from mlia.utils.logging import log_boundary_action


def install_shell_completion_with_notice(
    logger: logging.Logger, ctx: click.Context, param: click.Parameter, value: Any
) -> Any:
    """Announce shell configuration changes before Typer installs completion."""
    if not value or ctx.resilient_parsing:
        return value

    log_boundary_action(logger, "Installing shell completion.")
    return install_callback(ctx, param, value)
