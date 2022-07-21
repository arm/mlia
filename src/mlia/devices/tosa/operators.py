# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Operators module."""
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Protocol

from mlia.core._typing import PathOrFileLike


class TOSAChecker(Protocol):
    """TOSA checker protocol."""

    def is_tosa_compatible(self) -> bool:
        """Return true if model is TOSA compatible."""

    def _get_tosa_compatibility_for_ops(self) -> List[Any]:
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
    operators: List[Operator]


def get_tosa_compatibility_info(
    tflite_model_path: PathOrFileLike,
) -> TOSACompatibilityInfo:
    """Return list of the operators."""
    checker = get_tosa_checker(tflite_model_path)

    if checker is None:
        raise Exception(
            "TOSA checker is not available. "
            "Please make sure that 'tosa_checker' package is installed: "
            "pip install mlia[tosa]"
        )

    ops = [
        Operator(item.location, item.name, item.is_tosa_compatible)
        for item in checker._get_tosa_compatibility_for_ops()  # pylint: disable=protected-access
    ]

    return TOSACompatibilityInfo(checker.is_tosa_compatible(), ops)


def get_tosa_checker(tflite_model_path: PathOrFileLike) -> Optional[TOSAChecker]:
    """Return instance of the TOSA checker."""
    try:
        import tosa_checker as tc  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    checker = tc.TOSAChecker(str(tflite_model_path))
    return cast(TOSAChecker, checker)
