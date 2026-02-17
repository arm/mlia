# SPDX-FileCopyrightText: Copyright 2023,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA target module."""

from mlia.plugins.plugins import TargetPlugin
from mlia.target.registry import TargetInfo, TargetRegistry
from mlia.target.tosa.advisor import configure_and_get_tosa_advisor
from mlia.target.tosa.config import TOSAConfiguration


class TOSATargetPlugin(TargetPlugin):
    """TOSA Target Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: TargetRegistry) -> None:
        """Register the target with the registry."""
        registry.register(
            "tosa",
            TargetInfo(
                supported_backends=["tosa-checker"],
                default_backends=["tosa-checker"],
                advisor_factory_func=configure_and_get_tosa_advisor,
                target_profile_cls=TOSAConfiguration,
            ),
            pretty_name="TOSA",
        )
