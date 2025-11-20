# SPDX-FileCopyrightText: Copyright 2023, 2025 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tosa-checker reporters."""
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.tosa_checker.compat import Operator
from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.advice_generation import Advice
from mlia.core.reporting import CompoundReport
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.reporters import MetadataDisplay
from mlia.target.tosa.reporters import report_target
from mlia.target.tosa.reporters import report_tosa_compatibility
from mlia.target.tosa.reporters import report_tosa_errors
from mlia.target.tosa.reporters import report_tosa_exception
from mlia.target.tosa.reporters import report_tosa_operators
from mlia.target.tosa.reporters import tosa_formatters


def test_tosa_report_target() -> None:
    """Test function report_target()."""
    report = report_target(TOSAConfiguration.load_profile("tosa"))
    assert report.to_plain_text()


@pytest.mark.parametrize(
    "display_data, check_pkg_version",
    [
        ([Advice(messages=[])], False),
        (MetadataDisplay, True),
        (TOSAConfiguration(target="tosa"), False),
        (
            [
                Operator(
                    name="Alpha", location="sample/location/A", is_tosa_compatible=False
                ),
                Operator(
                    name="Beta", location="sample/location/B", is_tosa_compatible=False
                ),
            ],
            None,
        ),
        (TOSACompatibilityInfo(tosa_compatible=True, operators=[]), False),
        (TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE), False),
    ],
)
def test_tosa_formatters(
    display_data: Any,
    check_pkg_version: bool,
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
) -> None:
    """Test function tosa_formatters() with valid input."""
    mock_version = MagicMock()
    monkeypatch.setattr(
        "mlia.target.tosa.metadata.get_pkg_version",
        MagicMock(return_value=mock_version),
    )

    if inspect.isclass(display_data):
        display_data = display_data(test_tflite_model)

    formatter = tosa_formatters(display_data)
    report = formatter(display_data)
    if check_pkg_version:
        assert (
            display_data.data_dict["tosa-checker"]["tosa_checker_version"]
            == mock_version
        )

    assert isinstance(report, Report)


def test_tosa_formatters_invalid_data() -> None:
    """Test tosa_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        tosa_formatters(12)


def test_report_tosa_operators() -> None:
    """Test report_tosa_operators function."""
    assert isinstance(report_tosa_operators([]), Table)


def test_report_tosa_exception() -> None:
    """Test report_tosa_exception function."""
    assert isinstance(report_tosa_exception(None), NestedReport)


def test_report_tosa_errors() -> None:
    """Test report_tosa_errors function."""
    assert isinstance(report_tosa_errors(err=None), NestedReport)


def test_report_tosa_compatibility() -> None:
    """Test report_tosa_compatibility function."""
    compat_info = TOSACompatibilityInfo(tosa_compatible=True, operators=[])
    assert isinstance(report_tosa_compatibility(compat_info), CompoundReport)
