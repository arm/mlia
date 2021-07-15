# Copyright 2021, Arm Ltd.
"""Tests for reports module."""
import sys
from contextlib import ExitStack as doesnt_raise
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List

import pytest
from mlia.metadata import NpuSupported
from mlia.metadata import Operation
from mlia.metrics import PerformanceMetrics
from mlia.reporters import Cell
from mlia.reporters import Column
from mlia.reporters import Format
from mlia.reporters import report
from mlia.reporters import report_operators
from mlia.reporters import report_perf_metrics
from mlia.reporters import Table
from typing_extensions import Literal


@pytest.mark.parametrize(
    "data, formatters",
    [
        (
            [Operation("test_operation", "test_type", NpuSupported(False, []))],
            [report_operators, None],
        ),
        (
            PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            [report_perf_metrics, None],
        ),
        (
            [
                (
                    [Operation("test_operation", "test_type", NpuSupported(False, []))],
                    PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
        [
            "txt",
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
    fmt: Literal["txt", "json", "csv"],
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
            report(data, formatter, fmt, output)

            if is_file:
                assert output.is_file()


def test_table_representation() -> None:
    """Test table report representation."""

    def sample_table() -> Table:
        columns = [
            Column("Header 1", alias="header1", only_for=["txt"]),
            Column("Header 2", alias="header2", fmt=Format(wrap_width=5)),
            Column("Header 3", alias="header3"),
        ]
        rows = [(1, 2, 3), (4, 5, Cell(123123, fmt=Format(str_fmt="10,d")))]

        return Table(columns, rows, name="Sample table", alias="sample_table")

    table = sample_table()
    csv_repr = table.to_csv()
    assert csv_repr == [["Header 2", "Header 3"], [2, 3], [5, 123123]]

    json_repr = table.to_json()
    assert json_repr == {
        "sample_table": [
            {"header2": 2, "header3": 3},
            {"header2": 5, "header3": 123123},
        ]
    }

    text_report = table.to_text()
    expected_text_report = """
╒════════════╤════════════╤════════════╕
│ Header 1   │ Header 2   │ Header 3   │
╞════════════╪════════════╪════════════╡
│ 1          │ 2          │ 3          │
├────────────┼────────────┼────────────┤
│ 4          │ 5          │ 123,123    │
╘════════════╧════════════╧════════════╛
""".strip()
    print(text_report)
    assert text_report == expected_text_report
