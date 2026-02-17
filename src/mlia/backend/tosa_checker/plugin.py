# SPDX-FileCopyrightText: Copyright 2023,2025-2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA checker backend module."""

from mlia.backend.config import BackendConfiguration, BackendType, System
from mlia.backend.registry import BackendRegistry
from mlia.backend.tosa_checker.install import get_tosa_backend_installation
from mlia.core.common import AdviceCategory
from mlia.plugins.plugins import BackendPlugin


class TOSACheckerBackendPlugin(BackendPlugin):
    """TOSA Checker Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
        registry.register(
            "tosa-checker",
            BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=[System.LINUX_AMD64],
                backend_type=BackendType.WHEEL,
                installation=get_tosa_backend_installation(),
            ),
            pretty_name="TOSA Checker",
        )
