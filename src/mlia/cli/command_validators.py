# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI command validators module."""

import logging
import sys

from mlia.core.check_validation import (
    normalize_string as normalize_string,
)
from mlia.core.check_validation import (
    validate_backend as validate_backend,
)
from mlia.core.check_validation import (
    validate_check_target_profile as validate_check_target_profile,
)

logger = logging.getLogger(__name__)


def validate_optimize_target_profile(target_profile: str) -> None:
    """Validate whether the provided target profile is compatible with 'mlia optimize'.

    This function exits with code 1 if the provided target profile is
    not supported.
    """
    incompatible_targets_optimize: list[str] = ["tosa"]
    if target_profile in incompatible_targets_optimize:
        logger.error(
            "Optimization cannot be performed with target profile %s.", target_profile
        )
        sys.exit(1)
