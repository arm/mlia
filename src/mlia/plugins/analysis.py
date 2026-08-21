# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Analysis plugin interfaces.

Analysis plugins add optional post-analysis behaviour to MLIA commands. They may
provide Click options that are registered before command-line parsing and are
invoked after a successful analysis run with MLIA's canonical structured output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import click

from mlia.core.context import ExecutionContext


@dataclass(frozen=True)
class AnalysisRunResult:
    """Result and context passed to enabled analysis plugins."""

    output: dict[str, Any] | None
    context: ExecutionContext
    args: Mapping[str, object]
    command_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        """Return the MLIA output directory for this run."""
        return self.context.output_dir


class AnalysisPlugin(Protocol):
    """Protocol implemented by analysis plugin instances."""

    name: str

    def cli_options(self) -> list[click.Option]:
        """Return options to register on the MLIA check command."""

    def enabled(self, args: Mapping[str, object]) -> bool:
        """Return whether this plugin should run for the parsed arguments."""

    def run(self, result: AnalysisRunResult) -> None:
        """Run post-analysis behaviour."""


class AnalysisPluginRegistry:
    """Named registry for analysis plugins."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self.items: dict[str, AnalysisPlugin] = {}
        self.plugin_interface_versions: dict[str, str | None] = {}

    @property
    def plugins(self) -> list[AnalysisPlugin]:
        """Return registered plugins in registration order."""
        return list(self.items.values())

    def register(self, plugin: AnalysisPlugin) -> bool:
        """Register a uniquely named analysis plugin."""
        if plugin.name in self.items:
            return False
        self.items[plugin.name] = plugin
        return True

    def cli_options(self) -> list[click.Option]:
        """Return command-line options from all registered plugins."""
        return [option for plugin in self.plugins for option in plugin.cli_options()]

    def enabled_plugins(self, args: Mapping[str, object]) -> list[AnalysisPlugin]:
        """Return plugins enabled by parsed CLI args."""
        return [plugin for plugin in self.plugins if plugin.enabled(args)]
