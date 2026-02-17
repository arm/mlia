# SPDX-FileCopyrightText: Copyright 2023,2025-2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela backend module."""

from mlia.backend.config import BackendConfiguration, BackendType, System
from mlia.backend.registry import BackendRegistry
from mlia.backend.vela.install import get_vela_installation
from mlia.core.common import AdviceCategory
from mlia.plugins.plugins import BackendPlugin


class VelaBackendPlugin(BackendPlugin):
    """Vela Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
        registry.register(
            "vela",
            BackendConfiguration(
                supported_advice=[
                    AdviceCategory.COMPATIBILITY,
                    AdviceCategory.PERFORMANCE,
                    AdviceCategory.OPTIMIZATION,
                ],
                supported_systems=[
                    System.LINUX_AMD64,
                    System.LINUX_AARCH64,
                    System.WINDOWS_AMD64,
                    System.WINDOWS_AARCH64,
                ],
                backend_type=BackendType.WHEEL,
                installation=get_vela_installation(),
            ),
            pretty_name="Vela",
        )
