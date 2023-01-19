# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tosa-checker reporters."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.reporting import Report
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.reporters import MetadataDisplay
from mlia.target.tosa.reporters import report_target
from mlia.target.tosa.reporters import tosa_formatters


def test_tosa_report_target() -> None:
    """Test function report_target()."""
    report = report_target(TOSAConfiguration.load_profile("tosa"))
    assert report.to_plain_text()


def test_tosa_formatters(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test function tosa_formatters() with valid input."""
    mock_version = MagicMock()
    monkeypatch.setattr(
        "mlia.target.tosa.metadata.get_pkg_version",
        MagicMock(return_value=mock_version),
    )

    display_data = MetadataDisplay(test_tflite_model)
    formatter = tosa_formatters(MetadataDisplay(test_tflite_model))
    report = formatter(display_data)
    assert display_data.data_dict["tosa-checker"]["tosa_version"] == mock_version
    assert isinstance(report, Report)


def test_tosa_formatters_invalid_data() -> None:
    """Test tosa_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        tosa_formatters(12)
