# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra reporters."""
import pytest

from mlia.core.reporting import Report
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.reporters import hydra_formatters
from mlia.target.hydra.reporters import report_target


def test_report_target() -> None:
    """Test function report_target()."""
    report = report_target(HydraConfiguration.load_profile("hydra"))
    assert report.to_plain_text()


def test_hydra_formatters() -> None:
    """Test function hydra_formatters() with valid input."""
    with pytest.raises(NotImplementedError):
        formatter = hydra_formatters({})
        report = formatter({})
        assert isinstance(report, Report)


def test_hydra_formatters_invalid_data() -> None:
    """Test hydra_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        hydra_formatters(12)
