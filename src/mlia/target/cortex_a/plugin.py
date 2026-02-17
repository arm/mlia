# SPDX-FileCopyrightText: Copyright 2023,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A target module."""

from mlia.plugins.plugins import TargetPlugin
from mlia.target.cortex_a.advisor import configure_and_get_cortexa_advisor
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.registry import TargetInfo, TargetRegistry


class CortexATargetPlugin(TargetPlugin):
    """Cortex-A Target Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: TargetRegistry) -> None:
        """Register the target with the registry."""
        registry.register(
            "cortex-a",
            TargetInfo(
                supported_backends=["armnn-tflite-delegate"],
                default_backends=["armnn-tflite-delegate"],
                advisor_factory_func=configure_and_get_cortexa_advisor,
                target_profile_cls=CortexAConfiguration,
            ),
            pretty_name="Cortex-A",
        )
