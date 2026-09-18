# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI settings helpers."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from dotenv import dotenv_values
from platformdirs import user_config_path
from rich.console import Console
from rich.theme import Theme

from mlia.core.errors import ConfigurationError
from mlia.core.settings import (
    ApplicationSettings,
    FilteringSettings,
    ThemeName,
    parse_filtering_settings,
)
from mlia.utils.misc import merge

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

STANDARD_BORDER_COLOR = "#B8BECB"
STANDARD_NAME_COLOR = "#6B9BF8"
STANDARD_VERSION_COLOR = "#9385F8"
STANDARD_HIGHLIGHT_COLOR = "#F69B4F"
STANDARD_REQUIRED_COLOR = "#ED564D"
LIGHT_BORDER_COLOR = "#40454F"
LIGHT_NAME_COLOR = "#0157FF"
LIGHT_VERSION_COLOR = "#6A36ED"
LIGHT_HIGHLIGHT_COLOR = "#DB8232"
LIGHT_REQUIRED_COLOR = "#B4181B"


class Unset:
    """A simple sentinel type for when None has a proper meaning."""

    pass


@dataclass(frozen=True)
class ThemeColors:
    """Semantic colors used by Rich and Typer output."""

    border: str
    name: str
    version: str
    highlight: str
    required: str


DARK_COLORS = ThemeColors(
    STANDARD_BORDER_COLOR,
    STANDARD_NAME_COLOR,
    STANDARD_VERSION_COLOR,
    STANDARD_HIGHLIGHT_COLOR,
    STANDARD_REQUIRED_COLOR,
)
LIGHT_COLORS = ThemeColors(
    LIGHT_BORDER_COLOR,
    LIGHT_NAME_COLOR,
    LIGHT_VERSION_COLOR,
    LIGHT_HIGHLIGHT_COLOR,
    LIGHT_REQUIRED_COLOR,
)
THEME_COLORS: dict[ThemeName, ThemeColors] = {
    "dark": DARK_COLORS,
    "light": LIGHT_COLORS,
}


def _rich_theme(colors: ThemeColors) -> Theme:
    """Create a Rich theme from semantic CLI colors."""
    return Theme(
        {
            "warning": colors.highlight,
            "tbl.title": f"bold {colors.border}",
            "tbl.header": f"bold {colors.border}",
            "tbl.border": colors.border,
            "tbl.name": f"bold {colors.name}",
            "tbl.version": colors.version,
            "tbl.highlight": colors.highlight,
        }
    )


DARK = _rich_theme(DARK_COLORS)
LIGHT = _rich_theme(LIGHT_COLORS)
STANDARD = DARK

NO_COLOR = Theme(
    {
        "warning": "",
        "tbl.title": "bold",
        "tbl.header": "bold",
        "tbl.border": "",
        "tbl.name": "",
        "tbl.version": "",
        "tbl.highlight": "",
    },
    inherit=False,
)

_CONFIG_PATH = user_config_path("mlia", appauthor="arm") / "config.toml"
_TOP_LEVEL_KEYS = frozenset(
    {"core", "filtering", "plugins", "color", "theme", "backend_options"}
)


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


def _theme_name(config: TOMLVerifier) -> ThemeName:
    """Resolve the CLI theme from the environment and configuration."""
    light = os.getenv("MLIA_LIGHT", "") != ""
    dark = os.getenv("MLIA_DARK", "") != ""
    if light and dark:
        raise ConfigurationError("MLIA_LIGHT and MLIA_DARK cannot both be set.")
    if light:
        return "light"
    if dark:
        return "dark"

    configured = config.get(str, "theme", "dark")
    if configured not in THEME_COLORS:
        raise ConfigurationError(
            'MLIA Configuration value "theme" must be "light" or "dark".'
        )
    return cast(ThemeName, configured)


def get_environment() -> dict[str, str | None]:
    """Load the environment variables in place."""
    return {**dotenv_values(), **os.environ}


def new_settings(
    *,
    source: ApplicationSettings | None = None,
    color: bool | None = None,
    backend_options: dict[str, Any] = {},
    core_settings: dict[str, Any] = {},
    filtering: FilteringSettings | None = None,
    plugin_settings: dict[str, dict[str, Any]] = {},
) -> ApplicationSettings:
    """Build an ApplicationSettings object reading from the config file."""
    color_: bool = False
    backend_options_: dict[str, Any] = {}
    core_: dict[str, Any] = {}
    filtering_ = FilteringSettings()
    plugins_: dict[str, dict[str, Any]] = {}

    config = _read_config() if source is None else _to_toml_verifier(source)
    theme_name = _theme_name(config)

    if color is None:
        if _color_enabled() is None:
            color_ = config.get(bool, "color", True)
        else:
            color_ = False
    else:
        color_ = bool(color)

    backend_options_ = merge(config.get(dict, "backend_options", {}), backend_options)

    core_ = merge(config.get(dict, "core", {}), core_settings)

    filtering_ = (
        filtering
        if filtering is not None
        else source.filtering
        if source is not None
        else parse_filtering_settings(config.get(dict, "filtering", {}))
    )

    plugins_ = merge(config.get(dict, "plugins", {}), plugin_settings)

    theme = _rich_theme(THEME_COLORS[theme_name]) if color_ else NO_COLOR
    return ApplicationSettings(
        console=Console(no_color=not color_, theme=theme),
        color=color_,
        theme=theme_name,
        backend_options=backend_options_,
        core_settings=core_,
        filtering=filtering_,
        plugin_settings=plugins_,
    )


@dataclass
class TOMLVerifier:
    """Helper class for verifying the types of TOML output."""

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, cls: type[T], name: str, default: T) -> T:
        """Read a an object and verify that is has the correct type."""
        if name not in self.data:
            return default

        obj = self.data[name]
        if not isinstance(obj, cls):
            raise ConfigurationError(
                f'MLIA Configuration value "{name}" is not of type {str(cls)}'
            )

        return obj


def _read_config() -> TOMLVerifier:
    if not _CONFIG_PATH.exists():
        return TOMLVerifier()

    try:
        with _CONFIG_PATH.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise ConfigurationError(
            f"Unable to load configuration file {_CONFIG_PATH}: {err}"
        ) from err

    unknown_keys = set(config) - _TOP_LEVEL_KEYS
    for k in unknown_keys:
        logger.warning("Unknown top-level MLIA configuration key: %s", k)

    return TOMLVerifier(config)


def _to_toml_verifier(settings: ApplicationSettings) -> TOMLVerifier:
    return TOMLVerifier(
        {
            "color": settings.color,
            "theme": settings.theme,
            "backend_options": settings.backend_options,
            "core": settings.core_settings,
            "plugins": settings.plugin_settings,
        }
    )


def _color_enabled() -> bool | None:
    """Return whether CLI colors should be enabled."""
    if not sys.stdout.isatty():
        return False

    if (
        os.getenv("NO_COLOR", "") == ""
        and os.getenv("MLIA_NO_COLOR", "") == ""
        and os.getenv("TERM", "") != "dumb"
    ):
        return None

    return False
