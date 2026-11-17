# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test the Registry base class."""

from mlia.utils.registry import Registry


def test_registry() -> None:
    """Test Registry class."""
    reg = Registry[str]()
    assert not str(reg)
    assert reg.names() == []
    assert reg.register("name", "value")
    assert reg.names() == ["name"]
    assert not reg.register("name", "value")
    assert "name" in reg.items
    assert reg.items["name"] == "value"
    assert str(reg)
    assert reg.register("other_name", "value_2")
    assert len(reg.items) == 2
    assert "other_name" in reg.items
    assert reg.items["other_name"] == "value_2"
    assert reg.names() == ["name", "other_name"]


def test_registry_tracks_plugin_interface_version() -> None:
    """Registry metadata should store plugin interface versions."""
    reg = Registry[str]()

    assert reg.register("name", "value")
    reg.plugin_interface_versions["name"] = "0.0.2"

    assert reg.plugin_interface_versions["name"] == "0.0.2"
    assert reg.items["name"] == "value"


def test_registry_loads_once_on_use_and_allows_registration() -> None:
    """Importing a registry is inert; reentrant registration does not reload it."""
    calls = []

    def load() -> None:
        calls.append(True)
        registry.register("provider", "loaded")
        assert registry.items == {"provider": "loaded"}

    registry = Registry[str](loader=load)
    assert not calls
    assert registry.get("provider") == "loaded"
    assert registry.names() == ["provider"]
    assert calls == [True]


def test_replacing_items_before_discovery_preserves_replacement() -> None:
    """Providers must not be added back after an explicit registry replacement."""

    def load() -> None:
        registry.register("provider", "loaded")

    registry = Registry[str](loader=load)
    replacement = {"custom": "replacement"}
    registry.items = replacement
    assert registry.items is replacement
    assert registry.names() == ["custom"]


def test_registration_cannot_shadow_an_undiscovered_provider() -> None:
    """Duplicate registration has the same outcome before and after discovery."""

    def load() -> None:
        registry.register("provider", "loaded")

    registry = Registry[str](loader=load)
    assert not registry.register("provider", "replacement")
    assert registry.get("provider") == "loaded"
