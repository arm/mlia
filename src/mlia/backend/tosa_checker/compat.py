# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA compatibility module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Protocol

from mlia.backend.errors import BackendUnavailableError
from mlia.core.typing import PathOrFileLike


class TOSAChecker(Protocol):
    """TOSA checker protocol."""

    def is_tosa_compatible(self) -> bool:
        """Return true if model is TOSA compatible."""

    def _get_tosa_compatibility_for_ops(self) -> list[Any]:
        """Return list of operators."""


@dataclass
class Operator:
    """Operator's TOSA compatibility info."""

    location: str
    name: str
    is_tosa_compatible: bool


@dataclass
class TOSACompatibilityInfo:
    """Models' TOSA compatibility information."""

    tosa_compatible: bool
    operators: list[Operator]


def get_tosa_compatibility_info(
    tflite_model_path: PathOrFileLike,
) -> TOSACompatibilityInfo:
    """Return list of the operators."""
    checker = get_tosa_checker(tflite_model_path)

    if checker is None:
        raise BackendUnavailableError(
            "Backend tosa-checker is not available", "tosa-checker"
        )

    ops = [
        Operator(item.location, item.name, item.is_tosa_compatible)
        for item in checker._get_tosa_compatibility_for_ops()  # pylint: disable=protected-access
    ]

    return TOSACompatibilityInfo(checker.is_tosa_compatible(), ops)


def get_tosa_checker(tflite_model_path: PathOrFileLike) -> TOSAChecker | None:
    """Return instance of the TOSA checker."""
    try:
        import tosa_checker as tc  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    checker = tc.TOSAChecker(str(tflite_model_path))
    return cast(TOSAChecker, checker)
