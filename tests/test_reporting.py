# Copyright 2021, Arm Ltd.
"""Tests for reporting module."""
from typing import List

import pandas as pd
import pytest
from mlia.reporting import BytesCell
from mlia.reporting import Cell
from mlia.reporting import ClockCell
from mlia.reporting import Column
from mlia.reporting import CyclesCell
from mlia.reporting import Format
from mlia.reporting import NestedReport
from mlia.reporting import ReportDataFrame
from mlia.reporting import ReportItem
from mlia.reporting import Table


@pytest.mark.parametrize(
    "cell, expected_repr",
    [
        (BytesCell(None), ""),
        (BytesCell(0), "0 bytes"),
        (BytesCell(1), "1 byte"),
        (BytesCell(100000), "100,000 bytes"),
        (ClockCell(None), ""),
        (ClockCell(0), "0 Hz"),
        (ClockCell(1), "1 Hz"),
        (ClockCell(100000), "100,000 Hz"),
        (CyclesCell(None), ""),
        (CyclesCell(0), "0 cycles"),
        (CyclesCell(1), "1 cycle"),
        (CyclesCell(100000), "100,000 cycles"),
    ],
)
def test_predefined_cell_types(cell: Cell, expected_repr: str) -> None:
    """Test predefined cell types."""
    assert str(cell) == expected_repr


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


@pytest.mark.parametrize(
    "report, expected_plain_text, expected_json_data, expected_csv_data",
    [
        (
            NestedReport(
                "Sample report",
                "sample_report",
                [
                    ReportItem("Item", "item", "item_value"),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
""".strip(),
            {
                "sample_report": {"item": "item_value"},
            },
            [
                ("item",),
                ("item_value",),
            ],
        ),
        (
            NestedReport(
                "Sample report",
                "sample_report",
                [
                    ReportItem(
                        "Item",
                        "item",
                        "item_value",
                        [ReportItem("Nested item", "nested_item", "nested_item_value")],
                    ),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
    Nested item                                      nested_item_value
""".strip(),
            {
                "sample_report": {
                    "item": {"nested_item": "nested_item_value"},
                },
            },
            [
                ("item", "nested_item"),
                ("item_value", "nested_item_value"),
            ],
        ),
    ],
)
def test_nested_report_representation(
    report: NestedReport,
    expected_plain_text: str,
    expected_json_data: dict,
    expected_csv_data: List,
) -> None:
    """Test representation of the NestedReport."""
    plain_text = report.to_plain_text()
    assert plain_text == expected_plain_text

    json_data = report.to_json()
    assert json_data == expected_json_data

    csv_data = report.to_csv()
    assert csv_data == expected_csv_data
