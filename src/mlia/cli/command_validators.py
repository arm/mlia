# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI command validators module."""
from __future__ import annotations

import argparse
import logging
import sys

from mlia.target.registry import default_backends
from mlia.target.registry import get_target
from mlia.target.registry import supported_backends

logger = logging.getLogger(__name__)


def validate_backend(
    target_profile: str, backend: list[str] | None
) -> list[str] | None:
    """Validate backend with given target profile.

    This validator checks whether the given target-profile and backend are
    compatible with each other.
    It assumes that prior checks where made on the validity of the target-profile.
    """
    target = get_target(target_profile)

    if not backend:
        return default_backends(target)

    compatible_backends = list(map(normalize_string, supported_backends(target)))
    backends = {normalize_string(b): b for b in backend}

    incompatible_backends = [b for b in backends if b not in compatible_backends]
    # Throw an error if any unsupported backends are used
    if incompatible_backends:
        raise argparse.ArgumentError(
            None,
            f"Backend {', '.join(backends[b] for b in incompatible_backends)} "
            f"not supported with target-profile {target_profile}.",
        )
    return backend


def validate_check_target_profile(target_profile: str, category: set[str]) -> None:
    """Validate whether advice category is compatible with the provided target_profile.

    This validator function raises warnings if any desired advice category is not
    compatible with the selected target profile. If no operation can be
    performed as a result of the validation, MLIA exits with error code 0.
    """
    incompatible_targets_performance: list[str] = ["tosa", "cortex-a"]
    incompatible_targets_compatibility: list[str] = []

    # Check which check operation should be performed
    try_performance = "performance" in category
    try_compatibility = "compatibility" in category

    # Cross check which of the desired operations can be performed on given
    # target-profile
    do_performance = (
        try_performance and target_profile not in incompatible_targets_performance
    )
    do_compatibility = (
        try_compatibility and target_profile not in incompatible_targets_compatibility
    )

    # Case: desired operations can be performed with given target profile
    if (try_performance == do_performance) and (try_compatibility == do_compatibility):
        return

    warning_message = "\nWARNING: "
    # Case: performance operation to be skipped
    if try_performance and not do_performance:
        warning_message += (
            "Performance checks skipped as they cannot be "
            f"performed with target profile {target_profile}."
        )

    # Case: compatibility operation to be skipped
    if try_compatibility and not do_compatibility:
        warning_message += (
            "Compatibility checks skipped as they cannot be "
            f"performed with target profile {target_profile}."
        )

    # Case: at least one operation will be performed
    if do_compatibility or do_performance:
        logger.warning(warning_message)
        return

    # Case: no operation will be performed
    warning_message += " No operation was performed."
    logger.warning(warning_message)
    sys.exit(0)


def normalize_string(value: str) -> str:
    """Given a string return the normalized version.

    E.g. Given "ToSa-cHecker" -> "tosachecker"
    """
    return value.lower().replace("-", "")
