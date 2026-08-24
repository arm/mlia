# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Settings for a running MLIA application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rich.console import Console

from mlia.core.errors import ConfigurationError


@dataclass(frozen=True)
class CollapseRule:
    """Select entities whose string attribute matches configured globs."""

    kind: str
    attribute: str
    globs: tuple[str, ...]


@dataclass(frozen=True)
class FilteringSettings:
    """Core standardized-output filtering settings."""

    collapse: tuple[CollapseRule, ...] = ()


@dataclass(frozen=True)
class ApplicationSettings:
    """Resolved settings for a running MLIA application."""

    console: Console = field(default_factory=Console)
    color: bool = True
    backend_options: dict[str, Any] = field(default_factory=dict)
    core_settings: dict[str, Any] = field(default_factory=dict)
    filtering: FilteringSettings = field(default_factory=FilteringSettings)
    plugin_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_plugin(self, name: str) -> Mapping[str, Any]:
        """Return the settings table owned by one plugin."""
        return self.plugin_settings.get(name, {})


def parse_filtering_settings(settings: Mapping[str, Any]) -> FilteringSettings:
    """Validate the core-owned filtering table."""
    unknown_keys = sorted(set(settings) - {"collapse"})
    if unknown_keys:
        raise ConfigurationError(
            "Unknown filtering setting(s): " + ", ".join(unknown_keys)
        )
    if "collapse" not in settings:
        return FilteringSettings()

    collapse = settings["collapse"]
    if not isinstance(collapse, list):
        raise ConfigurationError(
            "Filtering setting 'collapse' must be an array of tables."
        )

    rules: list[CollapseRule] = []
    for index, raw_rule in enumerate(collapse):
        name = f"filtering.collapse[{index}]"
        if not isinstance(raw_rule, dict):
            raise ConfigurationError(
                f"MLIA configuration value '{name}' must be a table."
            )
        unknown_rule_keys = sorted(set(raw_rule) - {"kind", "attribute", "globs"})
        if unknown_rule_keys:
            raise ConfigurationError(
                f"Unknown {name} setting(s): " + ", ".join(unknown_rule_keys)
            )
        kind = raw_rule.get("kind")
        attribute = raw_rule.get("attribute")
        globs = raw_rule.get("globs")
        if not isinstance(kind, str) or not kind:
            raise ConfigurationError(
                f"Filtering setting '{name}.kind' must be a non-empty string."
            )
        if not isinstance(attribute, str) or not attribute:
            raise ConfigurationError(
                f"Filtering setting '{name}.attribute' must be a non-empty string."
            )
        if (
            not isinstance(globs, list)
            or not globs
            or not all(isinstance(pattern, str) and pattern for pattern in globs)
        ):
            raise ConfigurationError(
                f"Filtering setting '{name}.globs' must be a non-empty array of "
                "non-empty strings."
            )
        rules.append(CollapseRule(kind, attribute, tuple(globs)))
    return FilteringSettings(collapse=tuple(rules))
