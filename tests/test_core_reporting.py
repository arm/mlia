# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reporting module."""
from __future__ import annotations

import pytest

from mlia.core.reporting import BytesCell
from mlia.core.reporting import Cell
from mlia.core.reporting import ClockCell
from mlia.core.reporting import Column
from mlia.core.reporting import CyclesCell
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import ReportItem
from mlia.core.reporting import resolve_output_format
from mlia.core.reporting import SingleRow
from mlia.core.reporting import Table
from mlia.core.typing import OutputFormat
from mlia.core.typing import PathOrFileLike
from mlia.utils.console import remove_ascii_codes


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


@pytest.mark.parametrize(
    "with_notes, expected_text_report",
    [
        [
            True,
            """
Sample table:
┌──────────┬──────────┬──────────┐
│ Header 1 │ Header 2 │ Header 3 │
╞══════════╪══════════╪══════════╡
│ 1        │ 2        │ 3        │
├──────────┼──────────┼──────────┤
│ 4        │ 5        │ 123,123  │
└──────────┴──────────┴──────────┘
Sample notes
    """.strip(),
        ],
        [
            False,
            """
Sample table:
┌──────────┬──────────┬──────────┐
│ Header 1 │ Header 2 │ Header 3 │
╞══════════╪══════════╪══════════╡
│ 1        │ 2        │ 3        │
├──────────┼──────────┼──────────┤
│ 4        │ 5        │ 123,123  │
└──────────┴──────────┴──────────┘
    """.strip(),
        ],
    ],
)
def test_table_representation(with_notes: bool, expected_text_report: str) -> None:
    """Test table report representation."""

    def sample_table(with_notes: bool) -> Table:
        columns = [
            Column("Header 1", alias="header1", only_for=["plain_text"]),
            Column("Header 2", alias="header2", fmt=Format(wrap_width=5)),
            Column("Header 3", alias="header3"),
        ]
        rows = [(1, 2, 3), (4, 5, Cell(123123, fmt=Format(str_fmt="0,d")))]

        return Table(
            columns,
            rows,
            name="Sample table",
            alias="sample_table",
            notes="Sample notes" if with_notes else None,
        )

    table = sample_table(with_notes)
    json_repr = table.to_json()
    assert json_repr == {
        "sample_table": [
            {"header2": 2, "header3": 3},
            {"header2": 5, "header3": 123123},
        ]
    }

    text_report = remove_ascii_codes(table.to_plain_text())
    assert text_report == expected_text_report


@pytest.mark.parametrize(
    "report, expected_plain_text, expected_json_data",
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
                        [ReportItem("Nested item", "nested_item", BytesCell(10))],
                    ),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
    Nested item                                               10 bytes
""".strip(),
            {
                "sample_report": {
                    "item": {"nested_item": {"unit": "bytes", "value": 10}},
                },
            },
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
                        [
                            ReportItem(
                                "Nested item",
                                "nested_item",
                                Cell(
                                    10, fmt=Format(str_fmt=lambda x: f"{x} cell value")
                                ),
                            )
                        ],
                    ),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
    Nested item                                          10 cell value
""".strip(),
            {
                "sample_report": {
                    "item": {"nested_item": 10},
                },
            },
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
                        [
                            ReportItem(
                                "Nested item",
                                "nested_item",
                                Cell(
                                    10, fmt=Format(str_fmt=lambda x: f"{x} cell value")
                                ),
                            )
                        ],
                    ),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
    Nested item                                          10 cell value
""".strip(),
            {
                "sample_report": {
                    "item": {"nested_item": 10},
                },
            },
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
                        [
                            ReportItem("Nested item", "nested_item", Cell(10)),
                        ],
                    ),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
    Nested item                                                     10
""".strip(),
            {
                "sample_report": {
                    "item": {"nested_item": 10},
                },
            },
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
                        [
                            ReportItem(
                                "Nested item", "nested_item", Cell(10, fmt=Format())
                            ),
                        ],
                    ),
                ],
            ),
            """
Sample report:
  Item                                                      item_value
    Nested item                                                     10
""".strip(),
            {
                "sample_report": {
                    "item": {"nested_item": 10},
                },
            },
        ),
    ],
)
def test_nested_report_representation(
    report: NestedReport,
    expected_plain_text: str,
    expected_json_data: dict,
) -> None:
    """Test representation of the NestedReport."""
    plain_text = report.to_plain_text()
    assert plain_text == expected_plain_text

    json_data = report.to_json()
    assert json_data == expected_json_data


def test_single_row_representation() -> None:
    """Test representation of the SingleRow."""
    single_row = SingleRow(
        columns=[
            Column("column1", "column1"),
        ],
        rows=[("value1", "value2")],
        name="Single row example",
        alias="simple_row_example",
    )

    expected_text = """
Single row example:
  column1                                               value1
""".strip()
    assert single_row.to_plain_text() == expected_text
    assert single_row.to_json() == {"simple_row_example": [{"column1": "value1"}]}

    with pytest.raises(Exception, match="Table should have only one row"):
        wrong_single_row = SingleRow(
            columns=[
                Column("column1", "column1"),
            ],
            rows=[
                ("value1", "value2"),
                ("value1", "value2"),
            ],
            name="Single row example",
            alias="simple_row_example",
        )
        wrong_single_row.to_plain_text()


@pytest.mark.parametrize(
    "output, expected_output_format",
    [
        [None, "plain_text"],
        ["", "plain_text"],
        ["some_file", "plain_text"],
        ["some_format.some_ext", "plain_text"],
        ["output.json", "json"],
    ],
)
def test_resolve_output_format(
    output: PathOrFileLike | None, expected_output_format: OutputFormat
) -> None:
    """Test function resolve_output_format."""
    assert resolve_output_format(output) == expected_output_format
