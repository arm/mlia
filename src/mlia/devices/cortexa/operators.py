# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A tools module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Operator:
    """Cortex-A compatibility information of the operator."""

    name: str
    location: str
    is_cortex_a_compatible: bool


@dataclass
class CortexACompatibilityInfo:
    """Model's operators."""

    cortex_a_compatible: bool
    operators: list[Operator] | None = None


def get_cortex_a_compatibility_info(
    _model_path: Path,
) -> CortexACompatibilityInfo | None:
    """Return list of model's operators."""
    return None


def report() -> None:
    """Generate supported operators report."""
    raise Exception(
        "Generating a supported operators report is not "
        "currently supported with Cortex-A target profile."
    )
