# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra configuration."""
from __future__ import annotations

from typing import Any

from mlia.target.config import TargetProfile


class HydraConfiguration(TargetProfile):
    """Hydra configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Hydra target configuration."""
        target = kwargs["target"]
        super().__init__(target, kwargs.get("backend", {}))

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "hydra":
            raise ValueError(f"Wrong target {self.target} for Hydra configuration.")
