# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA compatibility module."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Protocol

from mlia.backend.errors import BackendUnavailableError
from mlia.core.typing import PathOrFileLike
from mlia.utils.logging import capture_raw_output


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
    exception: Exception | None = None
    errors: list[str] | None = None
    std_out: list[str] | None = None


def get_tosa_compatibility_info(
    tflite_model_path: PathOrFileLike,
) -> TOSACompatibilityInfo:
    """Return list of the operators."""
    # Capture the possible exception in running get_tosa_checker
    try:
        with capture_raw_output(sys.stdout) as std_output_pkg, capture_raw_output(
            sys.stderr
        ) as stderr_output_pkg:
            checker = get_tosa_checker(tflite_model_path)
    except Exception as exc:  # pylint: disable=broad-except
        return TOSACompatibilityInfo(
            tosa_compatible=False,
            operators=[],
            exception=exc,
            errors=None,
            std_out=None,
        )

    # Capture the possible BackendUnavailableError when tosa-checker is not available
    if checker is None:
        raise BackendUnavailableError(
            "Backend tosa-checker is not available", "tosa-checker"
        )

    # Capture the possible exception when checking ops compatibility
    try:
        with capture_raw_output(sys.stdout) as std_output_ops, capture_raw_output(
            sys.stderr
        ) as stderr_output_ops:
            ops = [
                Operator(item.location, item.name, item.is_tosa_compatible)
                for item in checker._get_tosa_compatibility_for_ops()  # pylint: disable=protected-access
            ]
    except Exception as exc:  # pylint: disable=broad-except
        return TOSACompatibilityInfo(
            tosa_compatible=False,
            operators=[],
            exception=exc,
            errors=None,
            std_out=None,
        )

    # Concatenate all possbile stderr/stdout
    stderr_output = stderr_output_pkg + stderr_output_ops
    std_output = std_output_pkg + std_output_ops

    return TOSACompatibilityInfo(
        tosa_compatible=checker.is_tosa_compatible(),
        operators=ops,
        exception=None,
        errors=stderr_output,
        std_out=std_output,
    )


def get_tosa_checker(tflite_model_path: PathOrFileLike) -> TOSAChecker | None:
    """Return instance of the TOSA checker."""
    try:
        import tosa_checker as tc  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    checker = tc.TOSAChecker(str(tflite_model_path))
    return cast(TOSAChecker, checker)
