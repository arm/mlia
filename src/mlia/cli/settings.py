# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI presentation and shared application configuration."""

from mlia.core.application_settings import THEME_COLORS
from mlia.core.application_settings import get_environment as get_environment
from mlia.core.application_settings import new_settings as new_settings
from mlia.core.settings import ThemeName


def configure_typer_help(color: bool, theme: ThemeName = "dark") -> None:
    """Configure Typer's Rich help to use the selected MLIA theme."""
    if not color:
        return

    from typer import rich_utils

    colors = THEME_COLORS[theme]
    rich_utils.STYLE_OPTION = f"bold {colors.name}"
    rich_utils.STYLE_SWITCH = f"bold {colors.name}"
    rich_utils.STYLE_NEGATIVE_OPTION = f"bold {colors.version}"
    rich_utils.STYLE_NEGATIVE_SWITCH = f"bold {colors.version}"
    rich_utils.STYLE_METAVAR = f"bold {colors.version}"
    rich_utils.STYLE_USAGE = colors.highlight
    rich_utils.STYLE_REQUIRED_SHORT = colors.required
    rich_utils.STYLE_REQUIRED_LONG = colors.required
    rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {colors.name}"
    rich_utils.STYLE_OPTIONS_PANEL_BORDER = colors.border
    rich_utils.STYLE_COMMANDS_PANEL_BORDER = colors.border
