# SPDX-FileCopyrightText: Copyright 2022-2023,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""

from mlia.backend.config import BackendConfiguration
from mlia.plugins.plugins import load_backend_plugins
from mlia.utils.registry import Registry

BackendRegistry = Registry[BackendConfiguration]

# All supported targets are required to be registered here.
registry = BackendRegistry()

_plugins_loaded = False


def _ensure_plugins_loaded() -> None:
    """Load backend plugins once before registry access."""
    global _plugins_loaded  # pylint: disable=global-statement
    if _plugins_loaded:
        return
    load_backend_plugins(registry)
    _plugins_loaded = True


def ensure_backend_plugins_loaded() -> None:
    """Public helper to ensure backend plugins are loaded."""
    _ensure_plugins_loaded()


def get_supported_backends() -> list:
    """Get a list of all backends supported by the backend manager."""
    _ensure_plugins_loaded()
    return sorted(list(registry.items.keys()))


def get_supported_systems() -> dict:
    """Get a list of all systems supported by the backend manager."""
    _ensure_plugins_loaded()
    return {
        backend: config.supported_systems for backend, config in registry.items.items()
    }
