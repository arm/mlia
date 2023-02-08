# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A reporters."""
from typing import Any

import pytest

from mlia.core.advice_generation import Advice
from mlia.core.reporting import Report
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.operators import Operator
from mlia.target.cortex_a.reporters import cortex_a_formatters
from mlia.target.cortex_a.reporters import report_target


def test_report_target() -> None:
    """Test function report_target()."""
    report = report_target(CortexAConfiguration.load_profile("cortex-a"))
    assert report.to_plain_text()


@pytest.mark.parametrize(
    "data",
    (
        [Advice(["Sample", "Advice"])],
        TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE),
        [
            Operator(
                name="Test",
                location="loc",
                support_type=Operator.SupportType.OP_NOT_SUPPORTED,
                activation_func=TFL_ACTIVATION_FUNCTION.NONE,
            )
        ],
    ),
)
def test_cortex_a_formatters(data: Any) -> None:
    """Test function cortex_a_formatters() with valid input."""
    formatter = cortex_a_formatters(data)
    report = formatter(data)
    assert isinstance(report, Report)


def test_cortex_a_formatters_invalid_data() -> None:
    """Test cortex_a_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        cortex_a_formatters(12)
