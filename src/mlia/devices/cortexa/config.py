# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A configuration."""
from __future__ import annotations

from mlia.devices.config import IPConfiguration
from mlia.utils.filesystem import get_profile


class CortexAConfiguration(IPConfiguration):  # pylint: disable=too-few-public-methods
    """Cortex-A configuration."""

    def __init__(self, target_profile: str) -> None:
        """Init Cortex-A target configuration."""
        target_data = get_profile(target_profile)

        target = target_data["target"]
        if target != "cortex-a":
            raise Exception(f"Wrong target {target} for Cortex-A configuration")
        super().__init__(target)
