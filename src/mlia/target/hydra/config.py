# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra configuration."""
from __future__ import annotations

from typing import Any
from typing import cast

from mlia.target.config import TargetProfile


class HydraConfiguration(TargetProfile):
    """Hydra configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Hydra target configuration."""
        target = kwargs["target"]
        super().__init__(target)

        # Load backend config(s) to be handled by the backend(s) later.
        self.backend_config = cast(dict, kwargs.get("backend", {}))

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "hydra":
            raise ValueError(f"Wrong target {self.target} for Hydra configuration.")
