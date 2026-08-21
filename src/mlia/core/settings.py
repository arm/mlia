# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Settings for a running MLIA application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rich.console import Console


@dataclass(frozen=True)
class ApplicationSettings:
    """Resolved settings for a running MLIA application."""

    console: Console = field(default_factory=Console)
    color: bool = True
    backend_options: dict[str, Any] = field(default_factory=dict)
    core_settings: dict[str, Any] = field(default_factory=dict)
    plugin_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_plugin(self, name: str) -> Mapping[str, Any]:
        """Return the settings table owned by one plugin."""
        return self.plugin_settings.get(name, {})
