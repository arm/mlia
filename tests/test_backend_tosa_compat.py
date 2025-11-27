# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA compatibility."""
from __future__ import annotations

import importlib
import sys
import warnings
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
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: str | Path
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
    *,
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: str | Path,
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


def test_backend_module_deprecation_warning() -> None:
    """Test that importing the backend module triggers a deprecation warning."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Re-import the backend module to trigger the warning
        # Use importlib to force a fresh import

        # Remove module from cache if it exists to force re-import
        module_name = "mlia.backend.tosa_checker"
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Import the module
        importlib.import_module(module_name)

        # Check that a deprecation warning was issued
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert any(
            deprecation_warnings
        ), "No DeprecationWarning was issued when importing backend module"

        # Check the warning message content
        warning_message = str(deprecation_warnings[0].message)
        assert "TOSA Checker backend is deprecated" in warning_message
        assert "unmaintained project" in warning_message


def test_tosa_compatibility_info_to_standardized_output(tmp_path: Path) -> None:
    """Test converting TOSACompatibilityInfo to standardized output."""
    # Create a test model file
    model_path = tmp_path / "test_model.tflite"
    model_path.write_bytes(b"test_model_content")

    # Create compatibility info
    operators = [
        Operator("loc1", "Conv2D", True),
        Operator("loc2", "Add", False),
    ]
    compat_info = TOSACompatibilityInfo(
        tosa_compatible=False,
        operators=operators,
        exception=None,
        errors=["Warning: Add not compatible"],
        std_out=None,
    )

    # Convert to standardized output
    output = compat_info.to_standardized_output(
        model_path=model_path,
        run_id="test-run-id",
        timestamp="2025-01-01T00:00:00Z",
        cli_arguments=["--model", "test.tflite"],
    )

    # Validate structure
    assert output.schema_version == "1.0.0"
    assert output.run_id == "test-run-id"
    assert output.timestamp == "2025-01-01T00:00:00Z"
    assert output.tool.name == "mlia"
    assert output.model.name == "test_model.tflite"
    assert output.model.format == "tflite"
    assert len(output.backends) == 1
    assert output.backends[0].id == "tosa-checker"
    assert output.target.profile_name == "tosa"
    assert len(output.results) == 1

    # Check result details
    result = output.results[0]
    assert result.kind.value == "compatibility"
    assert result.status.value == "incompatible"
    assert len(result.checks) == 2
    assert result.checks[0].status.value == "pass"
    assert result.checks[1].status.value == "fail"
    assert len(result.entities) == 2
    assert result.entities[0].name == "Conv2D"
    assert result.entities[1].name == "Add"
    assert "Warning: Add not compatible" in result.errors


def test_tosa_compatibility_info_to_standardized_output_with_exception(
    tmp_path: Path,
) -> None:
    """Test converting TOSACompatibilityInfo with exception to standardized output."""
    model_path = tmp_path / "test.tflite"
    model_path.write_bytes(b"test")

    compat_info = TOSACompatibilityInfo(
        tosa_compatible=False,
        operators=[],
        exception=ValueError("Test error"),
        errors=None,
        std_out=None,
    )

    output = compat_info.to_standardized_output(model_path=model_path)

    result = output.results[0]
    assert result.status.value == "failed"
    assert len(result.errors) == 1
    assert "Test error" in result.errors[0]
