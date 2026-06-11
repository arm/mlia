# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI command validators module."""

from __future__ import annotations

import logging
import sys

from mlia.core.errors import ConfigurationError
from mlia.target.registry import default_backends, get_target, supported_backends

logger = logging.getLogger(__name__)


def validate_backend(target_profile: str, backend: list[str] | None) -> list[str]:
    """Validate backend with given target profile.

    This validator checks whether the given target profile and backend are
    compatible with each other.
    It assumes that prior checks were made on the validity of the target profile.
    """
    target = get_target(target_profile)

    if not backend:
        return default_backends(target)

    compatible_backends = {
        normalize_string(canonical_backend): canonical_backend
        for canonical_backend in supported_backends(target)
    }
    backends = {normalize_string(b): b for b in backend}

    incompatible_backends = [b for b in backends if b not in compatible_backends]
    if incompatible_backends:
        raise ConfigurationError(
            f"Backend {', '.join(backends[b] for b in incompatible_backends)} "
            f"not supported with target profile {target_profile}.",
        )
    return [compatible_backends[b] for b in backends]


def validate_check_target_profile(target_profile: str, category: set[str]) -> bool:
    """Validate whether the advice category is compatible with the target profile.

    Logs warnings when a requested advice category is incompatible with the
    selected target profile. Returns ``False`` when no check operation should
    be performed, allowing the CLI entry point to decide how to exit.
    """
    incompatible_targets_performance: list[str] = ["tosa"]
    incompatible_targets_compatibility: list[str] = []

    # Check which check operation should be performed
    try_performance = "performance" in category
    try_compatibility = "compatibility" in category

    # Cross-check which of the desired operations can be performed on the given
    # target profile.
    do_performance = (
        try_performance and target_profile not in incompatible_targets_performance
    )
    do_compatibility = (
        try_compatibility and target_profile not in incompatible_targets_compatibility
    )

    # Case: desired operations can be performed with given target profile
    if (try_performance == do_performance) and (try_compatibility == do_compatibility):
        return True

    warning_message = "\nWARNING: "
    # Case: performance operation to be skipped
    if try_performance and not do_performance:
        warning_message += (
            "Performance checks skipped as they cannot be "
            f"performed with target profile {target_profile}."
        )

    # Case: compatibility operation to be skipped
    if try_compatibility and not do_compatibility:  # pragma: no cover, defensive code
        warning_message += (
            "Compatibility checks skipped as they cannot be "
            f"performed with target profile {target_profile}."
        )

    # Case: at least one operation will be performed
    if do_compatibility or do_performance:
        logger.warning(warning_message)
        return True

    # Case: no operation will be performed
    warning_message += " No operation was performed."
    logger.warning(warning_message)
    return False


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


def normalize_string(value: str) -> str:
    """Given a string return the normalized version.

    E.g. Given "ToSa-cHecker" -> "tosachecker"
    """
    return value.lower().replace("-", "")
