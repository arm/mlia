# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA target configuration."""
import logging
import warnings
from typing import Any

from mlia.target.config import TargetProfile

logger = logging.getLogger(__name__)


class TOSAConfiguration(TargetProfile):
    """TOSA configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init configuration."""
        target = kwargs["target"]
        super().__init__(target)

        # Warn user about deprecated backend usage
        warnings.warn(
            "The TOSA Checker backend is deprecated. "
            "This backend relies on an unmaintained project.",
            DeprecationWarning,
            stacklevel=2,
        )

        logger.warning(
            "Using deprecated TOSA Checker backend. It is deprecated due to "
            "dependency on an unmaintained project."
        )

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()
        if self.target != "tosa":
            raise ValueError(f"Wrong target {self.target} for TOSA configuration.")
