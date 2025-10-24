# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Corstone backend module."""
from mlia.backend.corstone.install import get_corstone_installation
from mlia.backend.registry import registry

# List of mutually exclusive Corstone backends ordered by priority
CORSTONE_PRIORITY = {
    "Corstone-320": get_corstone_installation(corstone_name="corstone-320"),
    "Corstone-310": get_corstone_installation(corstone_name="corstone-310"),
    "Corstone-300": get_corstone_installation(corstone_name="corstone-300"),
}


def is_corstone_backend(backend_name: str) -> bool:
    """Check if backend belongs to Corstone."""
    return any(
        name in CORSTONE_PRIORITY
        for name in (backend_name, registry.pretty_name(backend_name))
    )
