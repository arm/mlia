# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Validate supplemental analysis artifacts without importing format readers."""

from pathlib import Path

from mlia.core.errors import ConfigurationError


def validate_companion_model(companion_model: str | Path | None) -> Path | None:
    """Normalize the optional companion model path."""
    if companion_model is None:
        return None
    path = Path(companion_model).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Companion model file does not exist: {path}")
    return path
