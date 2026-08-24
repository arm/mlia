# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI command validators module."""

from __future__ import annotations

import logging
import sys

from mlia.backend.registry import registry as backend_registry
from mlia.core.errors import ConfigurationError
from mlia.target.registry import default_backends, get_target, supported_backends

logger = logging.getLogger(__name__)


def validate_backend(
    target_profile: str,
    backend: list[str] | None,
    *,
    profiling_data: bool = False,
) -> list[str]:
    """Validate and select backends for a target profile and input source."""
    target = get_target(target_profile)
    if not backend and not profiling_data:
        return default_backends(target)

    target_backends = supported_backends(target)
    compatible_backends = {
        normalize_string(canonical_backend): canonical_backend
        for canonical_backend in target_backends
    }

    if backend:
        backends = {normalize_string(name): name for name in backend}
        incompatible_backends = [
            name for name in backends if name not in compatible_backends
        ]
        if incompatible_backends:
            incompatible_names = ", ".join(
                backends[name] for name in incompatible_backends
            )
            raise ConfigurationError(
                f"Backend {incompatible_names} not supported with target profile "
                f"{target_profile}.",
            )
        selected = [compatible_backends[name] for name in backends]
        if not profiling_data:
            return selected
        if len(selected) != 1:
            raise ConfigurationError(
                "--profiling-data requires exactly one --backend value."
            )
        backend_name = selected[0]
        if not backend_registry.items[backend_name].supports_profiling_data:
            raise ConfigurationError(
                f"Backend '{backend_name}' does not support profiling data."
            )
        return selected

    profiling_backends = [
        name
        for name in target_backends
        if name in backend_registry.items
        and backend_registry.items[name].supports_profiling_data
    ]
    default_profiling_backends = [
        name for name in default_backends(target) if name in profiling_backends
    ]
    candidates = default_profiling_backends or profiling_backends
    if not candidates:
        raise ConfigurationError(
            f"Target profile '{target_profile}' has no installed backend that "
            "supports profiling data."
        )
    if len(candidates) > 1:
        raise ConfigurationError(
            "Multiple backends support profiling data for target profile "
            f"'{target_profile}': {', '.join(sorted(candidates))}. Select one "
            "with --backend."
        )
    return candidates


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
