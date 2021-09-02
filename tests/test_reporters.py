# Copyright 2021, Arm Ltd.
"""Tests for reports module."""
import sys
from contextlib import ExitStack as doesnt_raise
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List

import pandas as pd
import pytest
from mlia.config import EthosU55
from mlia.metadata import NpuSupported
from mlia.metadata import Operator
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics
from mlia.reporters import Cell
from mlia.reporters import Column
from mlia.reporters import Format
from mlia.reporters import report
from mlia.reporters import report_dataframe
from mlia.reporters import report_operators
from mlia.reporters import report_perf_metrics
from mlia.reporters import ReportDataFrame
from mlia.reporters import Table
from typing_extensions import Literal


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
            report(data, formatter, fmt, output)

            if is_file:
                assert output.is_file()


@pytest.mark.parametrize("with_notes", [True, False])
def test_table_representation(with_notes: bool) -> None:
    """Test table report representation."""

    def sample_table(with_notes: bool) -> Table:
        columns = [
            Column("Header 1", alias="header1", only_for=["plain_text"]),
            Column("Header 2", alias="header2", fmt=Format(wrap_width=5)),
            Column("Header 3", alias="header3"),
        ]
        rows = [(1, 2, 3), (4, 5, Cell(123123, fmt=Format(str_fmt="10,d")))]

        return Table(
            columns,
            rows,
            name="Sample table",
            alias="sample_table",
            notes="Sample notes" if with_notes else None,
        )

    table = sample_table(with_notes)
    csv_repr = table.to_csv()
    assert csv_repr == [["Header 2", "Header 3"], [2, 3], [5, 123123]]

    json_repr = table.to_json()
    assert json_repr == {
        "sample_table": [
            {"header2": 2, "header3": 3},
            {"header2": 5, "header3": 123123},
        ]
    }

    text_report = table.to_plain_text()
    if with_notes:
        expected_text_report = """
Sample table:
╒════════════╤════════════╤════════════╕
│ Header 1   │ Header 2   │ Header 3   │
╞════════════╪════════════╪════════════╡
│ 1          │ 2          │ 3          │
├────────────┼────────────┼────────────┤
│ 4          │ 5          │ 123,123    │
╘════════════╧════════════╧════════════╛
Sample notes
    """.strip()
    else:
        expected_text_report = """
Sample table:
╒════════════╤════════════╤════════════╕
│ Header 1   │ Header 2   │ Header 3   │
╞════════════╪════════════╪════════════╡
│ 1          │ 2          │ 3          │
├────────────┼────────────┼────────────┤
│ 4          │ 5          │ 123,123    │
╘════════════╧════════════╧════════════╛
    """.strip()
    assert text_report == expected_text_report


@pytest.mark.parametrize(
    "with_index, title, columns_name, notes, expected_text_report",
    [
        (
            True,
            "Sample table",
            "Sample index column",
            "",
            """
Sample table:
╒═══════════════════════╤════════════╤════════════╤════════════╕
│ Sample index column   │ Header 1   │ Header 2   │ Header 3   │
╞═══════════════════════╪════════════╪════════════╪════════════╡
│ 0                     │ 1          │ 2          │ 3          │
├───────────────────────┼────────────┼────────────┼────────────┤
│ 1                     │ 4          │ 5.56       │ 123,123    │
╘═══════════════════════╧════════════╧════════════╧════════════╛
    """.strip(),
        ),
        (
            False,
            "Sample table",
            "",
            "",
            """
Sample table:
╒════════════╤════════════╤════════════╕
│ Header 1   │ Header 2   │ Header 3   │
╞════════════╪════════════╪════════════╡
│ 1          │ 2          │ 3          │
├────────────┼────────────┼────────────┤
│ 4          │ 5.56       │ 123,123    │
╘════════════╧════════════╧════════════╛
    """.strip(),
        ),
        (
            False,
            "Sample table",
            "",
            "Sample note",
            """
Sample table:
╒════════════╤════════════╤════════════╕
│ Header 1   │ Header 2   │ Header 3   │
╞════════════╪════════════╪════════════╡
│ 1          │ 2          │ 3          │
├────────────┼────────────┼────────────┤
│ 4          │ 5.56       │ 123,123    │
╘════════════╧════════════╧════════════╛
Sample note
    """.strip(),
        ),
    ],
)
def test_reportdataframe_representation(
    with_index: bool,
    title: str,
    columns_name: str,
    notes: str,
    expected_text_report: str,
) -> None:
    """Test dataframe report representation."""

    def sample_df() -> pd.DataFrame:
        sample_dict = {
            "Header 1": [1, 4],
            "Header 2": [2, 5.55555],
            "Header 3": [3, 123123],
        }

        df = pd.DataFrame.from_dict(sample_dict)

        return df

    df = sample_df()
    csv_repr = ReportDataFrame(df).to_csv()
    expected_csv_repr = [
        ["Header 1", "Header 2", "Header 3"],
        [1, 2, 3],
        [4, 5.55555, 123123],
    ]
    assert csv_repr == expected_csv_repr

    json_repr = ReportDataFrame(df).to_json()
    expected_json_repr = {
        "Header 1": {0: 1, 1: 4},
        "Header 2": {0: 2.0, 1: 5.55555},
        "Header 3": {0: 3, 1: 123123},
    }
    assert json_repr == expected_json_repr

    df.loc[:, "Header 2"] = df["Header 2"].map("{:.2f}".format)
    df.loc[:, "Header 3"] = df["Header 3"].map("{:,d}".format)

    text_report = ReportDataFrame(df).to_plain_text(
        title=title, columns_name=columns_name, notes=notes, showindex=with_index
    )

    assert expected_text_report == text_report


def test_csv_nested_table_representation() -> None:
    """Test representation of the nested tables in csv format."""

    def sample_table(num_of_cols: int) -> Table:
        columns = [
            Column("Header 1", alias="header1"),
            Column("Header 2", alias="header2"),
        ]

        rows = [
            (
                1,
                Table(
                    columns=[
                        Column(f"Nested column {i+1}") for i in range(num_of_cols)
                    ],
                    rows=[[f"value{i+1}" for i in range(num_of_cols)]],
                    name="Nested table",
                ),
            )
        ]

        return Table(columns, rows, name="Sample table", alias="sample_table")

    assert sample_table(num_of_cols=2).to_csv() == [
        ["Header 1", "Header 2"],
        [1, "value1;value2"],
    ]

    assert sample_table(num_of_cols=1).to_csv() == [
        ["Header 1", "Header 2"],
        [1, "value1"],
    ]

    assert sample_table(num_of_cols=0).to_csv() == [
        ["Header 1", "Header 2"],
        [1, ""],
    ]
