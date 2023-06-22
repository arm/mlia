# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
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
