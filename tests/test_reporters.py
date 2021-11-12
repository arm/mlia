# Copyright 2021, Arm Ltd.
"""Tests for reports module."""
# pylint: disable=too-many-arguments
import sys
from contextlib import ExitStack as doesnt_raise
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from typing import Literal

import pytest
from mlia.config import EthosU55
from mlia.metadata import NpuSupported
from mlia.metadata import Operator
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics
from mlia.reporters import produce_report
from mlia.reporters import report_dataframe
from mlia.reporters import report_operators
from mlia.reporters import report_perf_metrics


@pytest.mark.parametrize(
    "data, formatters",
    [
        (
            [Operator("test_operator", "test_type", NpuSupported(False, []))],
            [report_operators, None],
        ),
        (
            PerformanceMetrics(
                EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
            ),
            [report_perf_metrics, None],
        ),
        (
            PerformanceMetrics(
                EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
            ).to_df(),
            [report_dataframe, None],
        ),
        (
            [
                (
                    [Operator("test_operator", "test_type", NpuSupported(False, []))],
                    PerformanceMetrics(
                        EthosU55(),
                        NPUCycles(0, 0, 0, 0, 0, 0),
                        MemoryUsage(0, 0, 0, 0, 0),
                    ),
                )
            ],
            [None],
        ),
    ],
)
@pytest.mark.parametrize(
    "fmt, output, expected_error",
    [
        [
            "unknown_format",
            sys.stdout,
            pytest.raises(Exception, match="Unknown format unknown_format"),
        ],
        [
            "plain_text",
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
        [
            "plain_text",
            "report.txt",
            doesnt_raise(),
        ],
        [
            "json",
            "report.json",
            doesnt_raise(),
        ],
        [
            "csv",
            "report.csv",
            doesnt_raise(),
        ],
    ],
)
def test_report(
    data: Any,
    formatters: List[Callable],
    fmt: Literal["plain_text", "json", "csv"],
    output: Any,
    expected_error: Any,
    tmpdir: Any,
) -> None:
    """Test report function."""
    is_file = isinstance(output, str)
    if is_file:
        output = Path(tmpdir) / output

    for formatter in formatters:
        with expected_error:
            produce_report(data, formatter, fmt, output)

            if is_file:
                assert output.is_file()
