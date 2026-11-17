# SPDX-FileCopyrightText: Copyright 2022-2023,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""

from mlia.backend.config import BackendConfiguration
from mlia.plugins.plugins import load_backend_plugins
from mlia.utils.registry import Registry

BackendRegistry = Registry[BackendConfiguration]


# Backend providers are registered on first access.
registry: BackendRegistry = BackendRegistry(
    loader=lambda: load_backend_plugins(registry)
)


def ensure_backend_plugins_loaded() -> None:
    """Public helper to ensure backend plugins are loaded."""
    registry.ensure_loaded()


def get_supported_backends() -> list:
    """Get a list of all backends supported by the backend manager."""
    return registry.names()


def get_selectable_backends() -> list[str]:
    """Return selectable backend names regardless of installation state."""
    return [backend for backend, config in registry.items.items() if config.selectable]


def get_supported_systems() -> dict:
    """Get a list of all systems supported by the backend manager."""
    return {
        backend: config.supported_systems for backend, config in registry.items.items()
    }
