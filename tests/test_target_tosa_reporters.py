# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tosa-checker reporters."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.metadata import MLIAMetadata
from mlia.core.metadata import ModelMetadata
from mlia.core.reporting import Report
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.metadata import TOSAMetadata
from mlia.target.tosa.reporters import MetadataDisplay
from mlia.target.tosa.reporters import report_device
from mlia.target.tosa.reporters import tosa_formatters


def test_tosa_report_device() -> None:
    """Test function report_device()."""
    report = report_device(TOSAConfiguration.load_profile("tosa"))
    assert report.to_plain_text()


def test_tosa_formatters(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test function tosa_formatters() with valid input."""
    mock_version = MagicMock()
    monkeypatch.setattr(
        "mlia.core.metadata.get_pkg_version",
        MagicMock(return_value=mock_version),
    )

    data = MetadataDisplay(
        TOSAMetadata("tosa-checker"),
        MLIAMetadata("mlia"),
        ModelMetadata(test_tflite_model),
    )
    formatter = tosa_formatters(data)
    report = formatter(data)
    assert data.tosa_version == mock_version
    assert isinstance(report, Report)


def test_tosa_formatters_invalid_data() -> None:
    """Test tosa_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        tosa_formatters(12)
