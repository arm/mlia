# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for analysis plugin support."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest

import mlia.plugins.plugins as plugin_utils
from mlia.plugins.analysis import AnalysisPluginRegistry, AnalysisRunResult


class DemoAnalysisPlugin:
    """Small analysis plugin used by tests."""

    name = "demo"

    def __init__(self) -> None:
        """Create the plugin."""
        self.results: list[AnalysisRunResult] = []

    def cli_options(self) -> list[click.Option]:
        """Return the test option."""
        return [click.Option(["--demo-analysis"], is_flag=True, default=False)]

    def enabled(self, args: Mapping[str, object]) -> bool:
        """Enable when the test option is set."""
        return bool(args.get("demo_analysis"))

    def run(self, result: AnalysisRunResult) -> None:
        """Record plugin execution."""
        self.results.append(result)


def test_analysis_plugin_registry_registers_options_and_filters_enabled() -> None:
    """Registry returns options and identifies enabled plugins."""
    registry = AnalysisPluginRegistry()
    plugin = DemoAnalysisPlugin()
    assert registry.register(plugin)
    assert not registry.register(plugin)
    assert registry.items == {"demo": plugin}

    options = registry.cli_options()

    assert len(options) == 1
    assert options[0].opts == ["--demo-analysis"]
    assert registry.enabled_plugins({"demo_analysis": True}) == [plugin]
    assert registry.enabled_plugins({"demo_analysis": False}) == []


def test_analysis_plugin_loader_records_interface_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared plugin loading records analysis plugin interface versions."""
    registry = AnalysisPluginRegistry()
    plugin = DemoAnalysisPlugin()
    module = SimpleNamespace(
        plugin_interface_version="0.0.2",
        register=lambda target: target.register(plugin),
    )
    monkeypatch.setattr(
        plugin_utils,
        "_load_plugin_modules",
        lambda group, supported_versions: iter([(MagicMock(), module)]),
    )

    plugin_utils.load_analysis_plugins(registry)

    assert registry.items == {"demo": plugin}
    assert registry.plugin_interface_versions == {"demo": "0.0.2"}


def test_analysis_run_result_exposes_output_dir() -> None:
    """AnalysisRunResult exposes the execution output directory."""
    ctx = MagicMock()
    ctx.output_dir = "out"

    result = AnalysisRunResult(
        output={"results": []},
        context=ctx,
        args={},
        command_name="check",
        parameters={"target_profile": "test"},
    )

    assert result.output_dir == "out"
    assert result.parameters == {"target_profile": "test"}
