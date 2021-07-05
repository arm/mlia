# Copyright 2021, Arm Ltd.
"""Tests for reports module."""
import sys
from contextlib import ExitStack as doesnt_raise
from typing import Any

import pytest
from mlia.metadata import NpuSupported
from mlia.metadata import Operation
from mlia.metrics import PerformanceMetrics
from mlia.reporters import report
from typing_extensions import Literal


@pytest.mark.parametrize(
    "data",
    [
        [Operation("test_operation", "test_type", NpuSupported(False, []))],
        PerformanceMetrics(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    ],
)
@pytest.mark.parametrize(
    "fmt, output, expected_error",
    [
        [
            "unknown_format",
            sys.stdout,
            pytest.raises(Exception, match="No reporter found"),
        ],
        [
            "txt",
            sys.stdout,
            doesnt_raise(),
        ],
        [
            "json",
            sys.stdout,
            doesnt_raise(),
        ],
        [
            "csv",
            sys.stdout,
            doesnt_raise(),
        ],
    ],
)
def test_report(
    data: Any, fmt: Literal["txt", "json", "csv"], output: Any, expected_error: Any
) -> None:
    """Test report function."""
    with expected_error:
        report(data, fmt, output)
