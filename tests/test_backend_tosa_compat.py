# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA compatibility."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.errors import BackendUnavailableError
from mlia.backend.tosa_checker.compat import get_tosa_compatibility_info
from mlia.backend.tosa_checker.compat import Operator
from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo


def replace_get_tosa_checker_with_mock(
    monkeypatch: pytest.MonkeyPatch, mock: MagicMock | None
) -> None:
    """Replace TOSA checker with mock."""
    monkeypatch.setattr(
        "mlia.backend.tosa_checker.compat.get_tosa_checker",
        MagicMock(return_value=mock),
    )


def test_compatibility_check_should_fail_if_checker_not_available(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test that compatibility check should fail if TOSA checker is not available."""
    replace_get_tosa_checker_with_mock(monkeypatch, None)

    with pytest.raises(
        BackendUnavailableError, match="Backend tosa-checker is not available"
    ):
        get_tosa_compatibility_info(test_tflite_model)


@pytest.mark.parametrize(
    "is_tosa_compatible, operators, exception, expected_result",
    [
        [
            True,
            [],
            None,
            TOSACompatibilityInfo(True, [], None, None, None),
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
            None,
            TOSACompatibilityInfo(
                True, [Operator("op_location", "op_name", True)], None, [], []
            ),
        ],
        [
            False,
            [],
            ValueError("error_test"),
            TOSACompatibilityInfo(False, [], ValueError("error_test"), [], []),
        ],
    ],
)
def test_get_tosa_compatibility_info(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    is_tosa_compatible: bool,
    operators: Any,
    exception: Exception | None,
    expected_result: TOSACompatibilityInfo,
) -> None:
    """Test getting TOSA compatibility information."""
    mock_checker = MagicMock()
    mock_checker.is_tosa_compatible.return_value = is_tosa_compatible
    mock_checker._get_tosa_compatibility_for_ops.return_value = (  # pylint: disable=protected-access
        operators
    )
    if exception:
        mock_checker._get_tosa_compatibility_for_ops.side_effect = (  # pylint: disable=protected-access
            exception
        )
    replace_get_tosa_checker_with_mock(monkeypatch, mock_checker)

    returned_compatibility_info = get_tosa_compatibility_info(test_tflite_model)
    assert repr(returned_compatibility_info.exception) == repr(
        expected_result.exception
    )
    assert (
        returned_compatibility_info.tosa_compatible == expected_result.tosa_compatible
    )
    assert returned_compatibility_info.operators == expected_result.operators
