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
