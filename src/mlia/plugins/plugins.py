# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Collection of plugin utilities."""

import logging
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

logger = logging.getLogger("mlia")

TARGET_PLUGIN_GROUP = "mlia.plugin.target"
BACKEND_PLUGIN_GROUP = "mlia.plugin.backend"

(MLIA_ENTRY_POINT,) = metadata.entry_points(group="console_scripts", name="mlia")

T = TypeVar("T")


class Plugin(ABC, Generic[T]):
    """Plugin definition class.

    Plugin 0.0.1 supports loading and exposing converters via the plugin interface.

    Attributes:
        plugin_interface_version - Compatible version of the plugin system.
    """

    plugin_interface_version: str

    @staticmethod
    @abstractmethod
    def register(registry: T) -> None:
        """Register plugin with associated registry."""


BackendPlugin = Plugin
TargetPlugin = Plugin


def call_entry_points(group: str, *args: Any) -> None:
    """Call all entry points of the given group with given args."""
    logger.debug("Loading plugins from '%s'", group)
    matching_entry_points = metadata.entry_points(group=group)
    for entry_point in matching_entry_points:
        if (
            entry_point.dist
            and MLIA_ENTRY_POINT.dist
            and entry_point.dist.name != MLIA_ENTRY_POINT.dist.name
        ):
            logger.debug(
                "Loading external plugin '%s' from '%s' (dist '%s')",
                entry_point.name,
                entry_point.value,
                entry_point.dist.name,
            )
        else:
            logger.debug(
                "Loading internal plugin '%s' from '%s'",
                entry_point.name,
                entry_point.value,
            )

        module = entry_point.load()

        if module.plugin_interface_version != "0.0.1":
            logger.error(
                "Incompatible version '%s' for plugin '%s'",
                module.plugin_interface_version,
                entry_point.name,
            )
            continue

        try:
            module.register(*args)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error("Error loading plugin '%s'", entry_point.name)
            logger.error(traceback.format_exc())


def load_target_plugins(*args: Any) -> None:
    """Load all target plugins by calling their entry points."""
    call_entry_points(TARGET_PLUGIN_GROUP, *args)


def load_backend_plugins(*args: Any) -> None:
    """Load all backend plugins by calling their entry points."""
    call_entry_points(BACKEND_PLUGIN_GROUP, *args)
