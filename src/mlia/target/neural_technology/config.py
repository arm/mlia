# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology configuration."""
from __future__ import annotations

from typing import Any

from mlia.target.config import TargetProfile


class NeuralTechnologyConfiguration(TargetProfile):
    """Neural Technology configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Neural Technology target configuration."""
        target = kwargs["target"]
        super().__init__(target, kwargs.get("backend", {}))

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "neural-technology":
            raise ValueError(
                f"Wrong target {self.target} for Neural Technology configuration."
            )
