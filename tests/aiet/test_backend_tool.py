# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use
"""Tests for the tool backend."""
from collections import Counter

import pytest

from aiet.backend.common import ConfigurationException
from aiet.backend.config import ToolConfig
from aiet.backend.tool import get_available_tool_directory_names
from aiet.backend.tool import get_available_tools
from aiet.backend.tool import get_tool
from aiet.backend.tool import Tool


def test_get_available_tool_directory_names() -> None:
    """Test get_available_tools mocking get_resources."""
    directory_names = get_available_tool_directory_names()
    assert Counter(directory_names) == Counter(["tool1", "tool2", "vela"])


def test_get_available_tools() -> None:
    """Test get_available_tools mocking get_resources."""
    available_tools = get_available_tools()
    expected_tool_names = sorted(
        [
            "tool_1",
            "tool_2",
            "vela",
            "vela",
            "vela",
        ]
    )

    assert all(isinstance(s, Tool) for s in available_tools)
    assert all(s != 42 for s in available_tools)
    assert any(s == available_tools[0] for s in available_tools)
    assert len(available_tools) == len(expected_tool_names)
    available_tool_names = sorted(str(s) for s in available_tools)
    assert available_tool_names == expected_tool_names


def test_get_tool() -> None:
    """Test get_tool mocking get_resoures."""
    tools = get_tool("tool_1")
    assert len(tools) == 1
    tool = tools[0]
    assert tool is not None
    assert isinstance(tool, Tool)
    assert tool.name == "tool_1"

    tools = get_tool("unknown tool")
    assert not tools


def test_tool_creation() -> None:
    """Test edge cases when creating a Tool instance."""
    with pytest.raises(ConfigurationException):
        Tool(ToolConfig(name="test", commands={"test": []}))  # no 'run' command
