# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A configuration."""
from __future__ import annotations

from typing import Any

from mlia.target.config import TargetProfile


class CortexAConfiguration(TargetProfile):
    """Cortex-A configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Cortex-A target configuration."""
        target = kwargs["target"]
        super().__init__(target)

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "cortex-a":
            raise ValueError(f"Wrong target {self.target} for Cortex-A configuration.")
