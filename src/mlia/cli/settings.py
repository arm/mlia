# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI settings helpers."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, TypeVar

from dotenv import dotenv_values
from platformdirs import user_config_path
from rich.console import Console
from rich.theme import Theme

from mlia.core.errors import ConfigurationError
from mlia.core.settings import (
    ApplicationSettings,
    FilteringSettings,
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


class Unset:
    """A simple sentinel type for when None has a proper meaning."""

    pass


STANDARD = Theme(
    {
        "warning": "yellow",
        "tbl.title": "bold dim",
        "tbl.header": "bold dim",
        "tbl.border": "dim",
        "tbl.name": "bold cyan",
        "tbl.highlight": "yellow",
    }
)

NO_COLOR = Theme(
    {
        "warning": "",
        "tbl.title": "bold",
        "tbl.header": "bold",
        "tbl.border": "",
        "tbl.name": "",
        "tbl.highlight": "",
    },
    inherit=False,
)

_CONFIG_PATH = user_config_path("mlia", appauthor="arm") / "config.toml"
_TOP_LEVEL_KEYS = frozenset(
    {"core", "filtering", "plugins", "color", "backend_options"}
)


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

    theme = STANDARD if color_ else NO_COLOR
    return ApplicationSettings(
        console=Console(no_color=not color_, theme=theme),
        color=color_,
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
