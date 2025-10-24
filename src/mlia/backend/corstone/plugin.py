# SPDX-FileCopyrightText: Copyright 2023,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Corstone backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.corstone import CORSTONE_PRIORITY
from mlia.backend.registry import BackendRegistry
from mlia.core.common import AdviceCategory
from mlia.plugins.plugins import BackendPlugin


class CorstoneBackendPlugin(BackendPlugin):
    """Corstone Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
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
