# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.utils.registry import Registry

# All supported targets are required to be registered here.
registry = Registry[BackendConfiguration]()


def get_supported_backends() -> list:
    """Get a list of all backends supported by the backend manager."""
    return sorted(list(registry.items.keys()))


def get_supported_systems() -> dict:
    """Get a list of all systems supported by the backend manager."""
    return {
        backend: config.supported_systems for backend, config in registry.items.items()
    }
