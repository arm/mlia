# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA target configuration."""
from typing import Any

from mlia.target.config import TargetProfile


class TOSAConfiguration(TargetProfile):
    """TOSA configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init configuration."""
        target = kwargs["target"]
        super().__init__(target)

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "tosa":
            raise ValueError(f"Wrong target {self.target} for TOSA configuration.")
