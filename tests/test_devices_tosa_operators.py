# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA compatibility."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Optional
from unittest.mock import MagicMock

import pytest

from mlia.devices.tosa.operators import get_tosa_compatibility_info
from mlia.devices.tosa.operators import Operator
from mlia.devices.tosa.operators import TOSACompatibilityInfo


def replace_get_tosa_checker_with_mock(
    monkeypatch: pytest.MonkeyPatch, mock: Optional[MagicMock]
) -> None:
    """Replace TOSA checker with mock."""
    monkeypatch.setattr(
        "mlia.devices.tosa.operators.get_tosa_checker", MagicMock(return_value=mock)
    )


def test_compatibility_check_should_fail_if_checker_not_available(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test that compatibility check should fail if TOSA checker is not available."""
    replace_get_tosa_checker_with_mock(monkeypatch, None)

    with pytest.raises(Exception, match="TOSA checker is not available"):
        get_tosa_compatibility_info(test_tflite_model)


@pytest.mark.parametrize(
    "is_tosa_compatible, operators, expected_result",
    [
        [
            True,
            [],
            TOSACompatibilityInfo(True, []),
        ],
        [
            True,
            [
                SimpleNamespace(
                    location="op_location",
                    name="op_name",
                    is_tosa_compatible=True,
                )
            ],
            TOSACompatibilityInfo(True, [Operator("op_location", "op_name", True)]),
        ],
        [
            False,
            [
                SimpleNamespace(
                    location="op_location",
                    name="op_name",
                    is_tosa_compatible=False,
                )
            ],
            TOSACompatibilityInfo(False, [Operator("op_location", "op_name", False)]),
        ],
    ],
)
def test_get_tosa_compatibility_info(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    is_tosa_compatible: bool,
    operators: Any,
    expected_result: TOSACompatibilityInfo,
) -> None:
    """Test getting TOSA compatibility information."""
    mock_checker = MagicMock()
    mock_checker.is_tosa_compatible.return_value = is_tosa_compatible
    mock_checker._get_tosa_compatibility_for_ops.return_value = (  # pylint: disable=protected-access
        operators
    )

    replace_get_tosa_checker_with_mock(monkeypatch, mock_checker)

    assert get_tosa_compatibility_info(test_tflite_model) == expected_result
