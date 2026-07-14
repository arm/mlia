# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for backend option discovery."""

from __future__ import annotations

import click
import pytest

from mlia.api import discover_backend_option_specs
from mlia.backend.config import BackendCliOption, BackendConfiguration, BackendType
from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory
from mlia.utils.registry import Registry


def _backend_config(
    cli_options: dict[str, BackendCliOption] | None = None,
) -> BackendConfiguration:
    """Return a backend config for option discovery tests."""
    return BackendConfiguration(
        supported_advice=[AdviceCategory.PERFORMANCE],
        supported_systems=None,
        backend_type=BackendType.CUSTOM,
        installation=None,
        cli_options=cli_options,
    )


def _record_plugin_interface_version(
    registry: Registry[BackendConfiguration],
    backend_name: str,
    plugin_interface_version: str,
) -> None:
    """Record plugin metadata for a registered backend."""
    registry.plugin_interface_versions[backend_name] = plugin_interface_version


@pytest.fixture(name="temporary_backend_registry")
def temporary_backend_registry_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> Registry[BackendConfiguration]:
    """Install and return a temporary backend registry for a test."""
    registry = Registry[BackendConfiguration]()
    monkeypatch.setattr(backend_registry, "items", registry.items)
    monkeypatch.setattr(
        backend_registry,
        "plugin_interface_versions",
        registry.plugin_interface_versions,
    )
    return registry


def test_discover_backend_option_specs_empty_registry(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """No registered backends means no backend option metadata."""
    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_skips_backends_without_cli_options(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Backends without declared CLI options should be ignored."""
    temporary_backend_registry.register("vela", _backend_config())

    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_extracts_cli_options(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Discovery should expose BackendConfiguration.cli_options as API metadata."""
    temporary_backend_registry.register(
        "bingo-bongo-backend",
        _backend_config(
            cli_options={
                "system_config": "--system-config",
                "compiler_config": "--compiler-config",
            },
        ),
    )
    _record_plugin_interface_version(
        temporary_backend_registry, "bingo-bongo-backend", "0.0.1"
    )

    specs = discover_backend_option_specs()

    assert specs == [
        {
            "module": "bingo_bongo_backend",
            "backend": "bingo-bongo-backend",
            "config_key": "system_config",
            "click_option": specs[0]["click_option"],
        },
        {
            "module": "bingo_bongo_backend",
            "backend": "bingo-bongo-backend",
            "config_key": "compiler_config",
            "click_option": specs[1]["click_option"],
        },
    ]
    first_click_option = specs[0]["click_option"]
    second_click_option = specs[1]["click_option"]
    assert isinstance(first_click_option, click.Option)
    assert first_click_option.opts == ["--bingo-bongo-backend.system-config"]
    assert first_click_option.name == "bingo_bongo_backend_system_config"
    assert isinstance(first_click_option.type, click.Path)
    assert isinstance(second_click_option, click.Option)
    assert second_click_option.opts == ["--bingo-bongo-backend.compiler-config"]
    assert second_click_option.name == "bingo_bongo_backend_compiler_config"
    assert isinstance(second_click_option.type, click.Path)


def test_discover_backend_option_specs_uses_legacy_path_type_for_version_001(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Legacy string backend options should keep the legacy Path conversion."""
    temporary_backend_registry.register(
        "legacy-backend",
        _backend_config(cli_options={"system_config": "--system-config"}),
    )
    _record_plugin_interface_version(
        temporary_backend_registry, "legacy-backend", "0.0.1"
    )

    specs = discover_backend_option_specs()

    assert specs == [
        {
            "module": "legacy_backend",
            "backend": "legacy-backend",
            "config_key": "system_config",
            "click_option": specs[0]["click_option"],
        }
    ]
    click_option = specs[0]["click_option"]
    assert isinstance(click_option, click.Option)
    assert click_option.opts == ["--legacy-backend.system-config"]
    assert click_option.name == "legacy_backend_system_config"
    assert isinstance(click_option.type, click.Path)


def test_discover_backend_option_specs_copies_and_namespaces_click_option(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Typed backend options should namespace a copy of the provided Click option."""
    optimization_level = click.Option(
        ["--optimization-level"],
        type=click.Choice(["0", "1", "2"]),
        help="Set optimization level.",
    )
    temporary_backend_registry.register(
        "typed-backend",
        _backend_config(cli_options={"optimization_level": optimization_level}),
    )
    _record_plugin_interface_version(
        temporary_backend_registry, "typed-backend", "0.0.2"
    )

    option_specs = discover_backend_option_specs()

    assert option_specs == [
        {
            "module": "typed_backend",
            "backend": "typed-backend",
            "config_key": "optimization_level",
            "click_option": option_specs[0]["click_option"],
        }
    ]
    click_option = option_specs[0]["click_option"]
    assert click_option is not optimization_level
    assert click_option.opts == ["--typed-backend.optimization-level"]
    assert click_option.name == "typed_backend_optimization_level"
    assert click_option.type is optimization_level.type
    assert click_option.help == "Set optimization level."
    assert optimization_level.opts == ["--optimization-level"]


def test_discover_backend_option_specs_rejects_short_typed_options(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Typed backend options should only expose namespaceable long options."""
    optimization_level = click.Option(
        ["-O", "--optimization-level"],
        type=click.Choice(["0", "1", "2"]),
    )
    temporary_backend_registry.register(
        "typed-backend",
        _backend_config(cli_options={"optimization_level": optimization_level}),
    )
    _record_plugin_interface_version(
        temporary_backend_registry, "typed-backend", "0.0.2"
    )

    with pytest.raises(TypeError, match="must only declare long options"):
        discover_backend_option_specs()


def test_discover_backend_option_specs_rejects_legacy_string_for_version_002(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Typed backend options should reject legacy string declarations."""
    temporary_backend_registry.register(
        "typed-backend",
        _backend_config(cli_options={"optimization_level": "--optimization-level"}),
    )
    _record_plugin_interface_version(
        temporary_backend_registry, "typed-backend", "0.0.2"
    )

    with pytest.raises(TypeError, match="Backend plugin 0.0.2 CLI option"):
        discover_backend_option_specs()


def test_discover_backend_option_specs_uses_plugin_version_for_option_shape(
    temporary_backend_registry: Registry[BackendConfiguration],
) -> None:
    """Backend option shape should be selected from the plugin interface version."""
    optimization_level = click.Option(
        ["-O", "--optimization-level"],
        type=click.Choice(["0", "1", "2"]),
    )
    temporary_backend_registry.register(
        "legacy-backend",
        _backend_config(cli_options={"optimization_level": optimization_level}),
    )
    _record_plugin_interface_version(
        temporary_backend_registry, "legacy-backend", "0.0.1"
    )

    with pytest.raises(TypeError, match="Backend plugin 0.0.1 CLI option"):
        discover_backend_option_specs()
