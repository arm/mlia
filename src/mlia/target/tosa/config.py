# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA target configuration."""
from mlia.target.config import IPConfiguration
from mlia.utils.filesystem import get_profile


class TOSAConfiguration(IPConfiguration):  # pylint: disable=too-few-public-methods
    """TOSA configuration."""

    def __init__(self, target_profile: str) -> None:
        """Init configuration."""
        target_data = get_profile(target_profile)
        target = target_data["target"]

        if target != "tosa":
            raise Exception(f"Wrong target {target} for TOSA configuration")

        super().__init__(target)
