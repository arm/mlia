# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tool backend module."""
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional

from aiet.backend.common import Backend
from aiet.backend.common import ConfigurationException
from aiet.backend.common import get_backend_configs
from aiet.backend.common import get_backend_directories
from aiet.backend.common import load_application_or_tool_configs
from aiet.backend.common import load_config
from aiet.backend.config import ExtendedToolConfig
from aiet.backend.config import ToolConfig


def get_available_tool_directory_names() -> List[str]:
    """Return a list of directory names for all available tools."""
    return [entry.name for entry in get_backend_directories("tools")]


def get_available_tools() -> List["Tool"]:
    """Return a list with all available tools."""
    available_tools = []
    for config_json in get_backend_configs("tools"):
        config_entries = cast(List[ExtendedToolConfig], load_config(config_json))
        for config_entry in config_entries:
            config_entry["config_location"] = config_json.parent.absolute()
            tools = load_tools(config_entry)
            available_tools += tools

    return sorted(available_tools, key=lambda tool: tool.name)


def get_tool(tool_name: str, system_name: Optional[str] = None) -> List["Tool"]:
    """Return a tool instance with the same name passed as argument."""
    return [
        tool
        for tool in get_available_tools()
        if tool.name == tool_name and (not system_name or tool.can_run_on(system_name))
    ]


def get_unique_tool_names(system_name: Optional[str] = None) -> List[str]:
    """Extract a list of unique tool names of all tools available."""
    return list(
        set(
            tool.name
            for tool in get_available_tools()
            if not system_name or tool.can_run_on(system_name)
        )
    )


class Tool(Backend):
    """Class for representing a single tool component."""

    def __init__(self, config: ToolConfig) -> None:
        """Construct a Tool instance from a dict."""
        super().__init__(config)

        self.supported_systems = config.get("supported_systems", [])

        if "run" not in self.commands:
            raise ConfigurationException("A Tool must have a 'run' command.")

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, Tool):
            return False

        return (
            super().__eq__(other)
            and self.name == other.name
            and set(self.supported_systems) == set(other.supported_systems)
        )

    def can_run_on(self, system_name: str) -> bool:
        """Check if the tool can run on the system passed as argument."""
        return system_name in self.supported_systems

    def get_details(self) -> Dict[str, Any]:
        """Return dictionary with all relevant information of the Tool instance."""
        output = {
            "type": "tool",
            "name": self.name,
            "description": self.description,
            "supported_systems": self.supported_systems,
            "commands": self._get_command_details(),
        }

        return output


def load_tools(config: ExtendedToolConfig) -> List[Tool]:
    """Load tool.

    Tool configuration could contain different parameters/commands for different
    supported systems. For each supported system this function will return separate
    Tool instance with appropriate configuration.
    """
    configs = load_application_or_tool_configs(
        config, ToolConfig, is_system_required=False
    )
    tools = [Tool(cfg) for cfg in configs]
    return tools
