# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Corstone backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.corstone.install import get_corstone_installation
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

# List of mutually exclusive Corstone backends ordered by priority

CORSTONE_PRIORITY = {
    "Corstone-310": get_corstone_installation(corstone_name="corstone-310"),
    "Corstone-300": get_corstone_installation(corstone_name="corstone-300"),
}


for corstone_name, installation in CORSTONE_PRIORITY.items():
    registry.register(
        corstone_name.lower(),
        BackendConfiguration(
            supported_advice=[
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.PERFORMANCE,
                AdviceCategory.OPTIMIZATION,
            ],
            supported_systems=[System.LINUX_AMD64, System.LINUX_AARCH64],
            backend_type=BackendType.CUSTOM,
            installation=installation,
        ),
        pretty_name=corstone_name,
    )


def is_corstone_backend(backend_name: str) -> bool:
    """Check if backend belongs to Corstone."""
    return any(
        name in CORSTONE_PRIORITY
        for name in (backend_name, registry.pretty_name(backend_name))
    )
