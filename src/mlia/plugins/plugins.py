# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Collection of plugin utilities."""

import logging
import sys
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from mlia.plugins.registry import check_core_compatibility
from mlia.utils.registry import Registry

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

logger = logging.getLogger("mlia")

TARGET_PLUGIN_GROUP = "mlia.plugin.target"
BACKEND_PLUGIN_GROUP = "mlia.plugin.backend"
CLI_PLUGIN_GROUP = "mlia.plugin.cli"
TRANSFORMER_PLUGIN_GROUP = "mlia.plugin.transformer"
SUPPORTED_PLUGIN_INTERFACE_VERSIONS = frozenset({"0.0.1", "0.0.2"})

(MLIA_ENTRY_POINT,) = metadata.entry_points(group="console_scripts", name="mlia")

T = TypeVar("T")


class Plugin(ABC, Generic[T]):
    """Plugin definition class.

    Plugin 0.0.1 supports loading and exposing converters via the plugin interface.
    Plugin 0.0.2 adds typed backend CLI options.

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


def _load_plugin_modules(group: str) -> Iterator[tuple[metadata.EntryPoint, Any]]:
    """Load compatible plugin modules from the given entry point group."""
    logger.debug("Loading plugins from '%s'", group)
    matching_entry_points = metadata.entry_points(group=group)
    for entry_point in matching_entry_points:
        is_internal = (
            entry_point.dist
            and MLIA_ENTRY_POINT.dist
            and entry_point.dist.name == MLIA_ENTRY_POINT.dist.name
        )
        if not is_internal and entry_point.dist and MLIA_ENTRY_POINT.dist:
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

        if not is_internal:
            is_compatible, error_message = check_core_compatibility(entry_point)
            if not is_compatible:
                logger.error("%s", error_message)
                continue

        try:
            module = entry_point.load()
        except Exception:
            logger.error("Error importing plugin '%s'", entry_point.name)
            logger.error(traceback.format_exc())
            continue

        if module.plugin_interface_version not in SUPPORTED_PLUGIN_INTERFACE_VERSIONS:
            logger.error(
                "Incompatible version '%s' for plugin '%s'",
                module.plugin_interface_version,
                entry_point.name,
            )
            continue

        yield entry_point, module


def _record_plugin_interface_version_for_new_items(
    registry: Registry[Any],
    previous_registry_names: set[str],
    plugin_interface_version: str,
) -> None:
    """Record plugin interface version for items added during registration."""
    new_names = {name for name in registry.items if name not in previous_registry_names}
    for name in new_names:
        registry.plugin_interface_versions[name] = plugin_interface_version


def call_entry_points(group: str, registry: Registry[Any]) -> None:
    """Call registry-backed entry points of the given group."""
    for entry_point, module in _load_plugin_modules(group):
        previous_registry_names = set(registry.items)

        try:
            module.register(registry)
        except Exception:
            logger.error("Error loading plugin '%s'", entry_point.name)
            logger.error(traceback.format_exc())
        finally:
            _record_plugin_interface_version_for_new_items(
                registry,
                previous_registry_names,
                module.plugin_interface_version,
            )


def load_target_plugins(registry: Registry[Any]) -> None:
    """Load all target plugins by calling their entry points."""
    call_entry_points(TARGET_PLUGIN_GROUP, registry)


def load_backend_plugins(registry: Registry[Any]) -> None:
    """Load all backend plugins by calling their entry points."""
    call_entry_points(BACKEND_PLUGIN_GROUP, registry)


def load_cli_plugins(registry: Registry[Any]) -> None:
    """Load all CLI plugins by calling their entry points."""
    call_entry_points(CLI_PLUGIN_GROUP, registry)


def load_transformer_plugins(registry: Registry[Any]) -> None:
    """Load all transformer plugins by calling their entry points."""
    call_entry_points(TRANSFORMER_PLUGIN_GROUP, registry)
