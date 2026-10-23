# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Validate supplemental analysis artifacts without importing format readers."""

from pathlib import Path

from mlia.core.errors import ConfigurationError


def validate_companion_model(
    model: object, companion_model: str | Path | None
) -> Path | None:
    """Normalize the optional deployed PTE companion of an ETRecord."""
    if companion_model is None:
        return None
    if not isinstance(model, (str, Path)) or Path(model).suffix.lower() not in {
        ".etrecord",
        ".bin",
    }:
        raise ConfigurationError("A companion model requires an ETRecord model.")
    path = Path(companion_model).expanduser().resolve()
    if path.suffix.lower() != ".pte":
        raise ConfigurationError("The companion model must reference a .pte file.")
    if not path.is_file():
        raise ConfigurationError(f"Companion model file does not exist: {path}")
    return path
