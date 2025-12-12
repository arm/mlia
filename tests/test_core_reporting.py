# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reporting module."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import mlia.core.output_schema as schema
from mlia.core.advice_generation import Advice
from mlia.core.reporting import BytesCell
from mlia.core.reporting import Cell
from mlia.core.reporting import ClockCell
from mlia.core.reporting import Column
from mlia.core.reporting import CustomJSONEncoder
from mlia.core.reporting import CyclesCell
from mlia.core.reporting import Format
from mlia.core.reporting import JSONReporter
from mlia.core.reporting import NestedReport
from mlia.core.reporting import ReportItem
from mlia.core.reporting import SingleRow
from mlia.core.reporting import Table
from mlia.core.reporting import TextReporter
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
    "value, nested_items, expected_compound, expected_raw_value",
    [
        (
            "value",
            [],
            False,
            "value",
        ),
        (
            Cell("value"),
            [],
            False,
            "value",
        ),
        (
            Cell("value"),
            [ReportItem("value")],
            True,
            "value",
        ),
    ],
)
def test_report_item(
    value: str | int | float | Cell | None,
    nested_items: list[ReportItem] | None,
    expected_compound: bool,
    expected_raw_value: Any,
) -> None:
    """Test ReportItem class."""
    report_item = ReportItem(
        "test_report_item", "report_item_alias", value, nested_items
    )
    assert report_item.compound == expected_compound
    assert report_item.raw_value == expected_raw_value


@pytest.mark.parametrize(
    "value, fmt, expected_str, expected_json",
    [
        (
            "value",
            None,
            "value",
            "value",
        ),
        (2.0, None, "2.0", 2.0),
        (2.111, Format(2, ".2f", "my_style"), "[my_style]2.11", 2.111),
    ],
)
def test_cell(value: Any, fmt: Format, expected_str: str, expected_json: Any) -> None:
    """Test Cell class."""
    cell = Cell(value, fmt)
    assert str(cell) == expected_str
    assert cell.to_json() == expected_json


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
                    ReportItem("Item", "item", None),
                ],
            ),
            """
Sample report:
  Item:
""".strip(),
            {
                "sample_report": {"item": None},
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


def test_custom_json_serialization() -> None:
    """Test JSON serialization for custom types."""

    class TestEnum(Enum):
        """Test enum."""

        VALUE1 = "value1"
        VALUE2 = "value2"

    table = Table(
        [Column("Column1", alias="column1")],
        rows=[[TestEnum.VALUE1], [np.float32(10.0)], [np.int64(10)]],
        name="sample_table",
        alias="sample_table",
    )

    output = json.dumps(table.to_json(), indent=4, cls=CustomJSONEncoder)

    assert json.loads(output) == {
        "sample_table": [
            {"column1": "value1"},
            {"column1": 10.0},
            {"column1": 10},
        ]
    }


def test_custom_json_encoder_with_dataclasses() -> None:
    """Test CustomJSONEncoder with dataclasses that have to_dict method."""
    output = schema.StandardizedOutput(
        schema_version=schema.SCHEMA_VERSION,
        run_id=schema.StandardizedOutput.create_run_id(),
        timestamp=schema.StandardizedOutput.create_timestamp(),
        tool=schema.Tool(name="mlia", version="1.0.0"),
        target=schema.Target(
            profile_name="test",
            target_type="ethos-u55",
            components=[
                schema.Component(type=schema.ComponentType.NPU, family="ethos-u")
            ],
            configuration={},
        ),
        model=schema.Model(name="test.tflite", format="tflite", hash="a" * 64),
        context=schema.Context(),
        backends=[
            schema.Backend(id="test", name="Test", version="1.0.0", configuration={})
        ],
        results=[],
    )

    # Test that CustomJSONEncoder can serialize the schema.StandardizedOutput
    serialized = json.dumps(output, cls=CustomJSONEncoder, indent=2)
    deserialized = json.loads(serialized)

    assert deserialized["schema_version"] == schema.SCHEMA_VERSION
    assert deserialized["tool"]["name"] == "mlia"
    assert deserialized["model"]["format"] == "tflite"


class TestTextReporter:
    """Test TextReporter methods."""

    def test_text_reporter(self) -> None:
        """Test TextReporter."""
        format_resolver = MagicMock()
        reporter = TextReporter(format_resolver)
        assert reporter.output_format == "plain_text"

    def test_submit(self) -> None:
        """Test TextReporter submit."""
        format_resolver = MagicMock()
        reporter = TextReporter(format_resolver)
        reporter.submit("test")
        assert reporter.data == [("test", ANY)]

        reporter.submit("test2", delay_print=True)
        assert reporter.data == [("test", ANY), ("test2", ANY)]
        assert reporter.delayed == [("test2", ANY)]

    def test_print_delayed(self) -> None:
        """Test TextReporter print_delayed."""
        with patch(
            "mlia.core.reporting.TextReporter.produce_report"
        ) as mock_produce_report:
            format_resolver = MagicMock()
            reporter = TextReporter(format_resolver)
            reporter.submit("test", delay_print=True)
            reporter.print_delayed()
            assert reporter.data == [("test", ANY)]
            assert not reporter.delayed
            mock_produce_report.assert_called()

    def test_produce_report(self) -> None:
        """Test TextReporter produce_report."""
        format_resolver = MagicMock()
        reporter = TextReporter(format_resolver)

        with patch("mlia.core.reporting.logger") as mock_logger:
            mock_formatter = MagicMock()
            reporter.produce_report("test", mock_formatter)
            mock_formatter.assert_has_calls([call("test"), call().to_plain_text()])
            mock_logger.info.assert_called()


class TestJSONReporter:
    """Test JSONReporter methods."""

    def test_text_reporter(self) -> None:
        """Test JSONReporter."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)
        assert reporter.output_format == "json"
        assert not reporter.standardized_outputs
        assert not reporter.advice_data

    def test_submit(self) -> None:
        """Test JSONReporter submit."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)
        reporter.submit("test")
        assert reporter.data == [("test", ANY)]
        assert not reporter.standardized_outputs

        reporter.submit("test2")
        assert reporter.data == [("test", ANY), ("test2", ANY)]
        assert not reporter.standardized_outputs

    def test_submit_with_standardized_output(self) -> None:
        """Test JSONReporter submit with standardized_output attribute."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)

        # Create mock object with standardized_output
        mock_data = MagicMock()
        mock_data.standardized_output = {"schema_version": "1.0.0"}

        reporter.submit(mock_data)
        assert len(reporter.data) == 1
        assert len(reporter.standardized_outputs) == 1
        assert reporter.standardized_outputs[0] == {"schema_version": "1.0.0"}

    def test_generate_report(self) -> None:
        """Test JSONReporter generate_report with legacy data."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)
        reporter.submit("test")

        with patch(
            "mlia.core.reporting.JSONReporter.produce_report"
        ) as mock_produce_report:
            reporter.generate_report()
            mock_produce_report.assert_called()

    def test_generate_report_with_standardized_output(self) -> None:
        """Test JSONReporter generate_report with standardized output."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)

        # Submit data with standardized_output
        mock_data = MagicMock()
        mock_data.standardized_output = {"schema_version": "1.0.0", "results": []}
        reporter.submit(mock_data)

        with patch(
            "mlia.core.reporting.JSONReporter._produce_standardized_report"
        ) as mock_standardized:
            reporter.generate_report()
            mock_standardized.assert_called_once()

    def test_generate_empty_report(self) -> None:
        """Test JSONReporter generate_report on empty data."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)
        with patch(
            "mlia.core.reporting.JSONReporter.produce_report"
        ) as mock_produce_report:
            reporter.generate_report()
            mock_produce_report.assert_not_called()

    @patch("builtins.print")
    def test_produce_report(self, mock_print: Mock) -> None:
        """Test JSONReporter produce_report."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)

        with patch("json.dumps") as mock_dumps:
            mock_formatter = MagicMock()
            reporter.produce_report("test", mock_formatter)
            mock_formatter.assert_has_calls([call("test"), call().to_json()])
            mock_dumps.assert_called()
            mock_print.assert_called()

    @patch("builtins.print")
    @patch("json.dumps")
    def test_produce_standardized_report(
        self, mock_dumps: Mock, _mock_print: Mock
    ) -> None:
        """Test JSONReporter _produce_standardized_report."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)

        # Add standardized output
        reporter.standardized_outputs = [{"schema_version": "1.0.0", "results": []}]

        reporter._produce_standardized_report()  # pylint: disable=protected-access

        mock_dumps.assert_called_once()
        # Verify it outputs the standardized format
        call_args = mock_dumps.call_args[0][0]
        assert call_args == {"schema_version": "1.0.0", "results": []}

    @patch("builtins.print")
    @patch("json.dumps")
    def test_produce_standardized_report_with_advice(
        self, mock_dumps: Mock, _mock_print: Mock
    ) -> None:
        """Test JSONReporter adds advice as extension."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)

        # Add standardized output
        reporter.standardized_outputs = [{"schema_version": "1.0.0", "results": []}]

        # Add advice with metadata
        advice = Advice(messages=["message1", "message2"], metadata=[{"key": "value"}])
        reporter.advice_data = [([advice], MagicMock())]

        reporter._produce_standardized_report()  # pylint: disable=protected-access

        # Verify advice was added to extensions with correct format
        call_args = mock_dumps.call_args[0][0]
        assert "extensions" in call_args
        assert "advice" in call_args["extensions"]
        advice_list = call_args["extensions"]["advice"]
        assert len(advice_list) == 1
        assert "id" in advice_list[0]
        assert advice_list[0]["message"] == "message1 message2"
        assert advice_list[0]["metadata"] == [{"key": "value"}]

    def test_merge_standardized_outputs(self) -> None:
        """Test merging multiple standardized outputs."""
        format_resolver = MagicMock()
        reporter = JSONReporter(format_resolver)

        outputs = [
            {
                "schema_version": "1.0.0",
                "run_id": "abc123",
                "results": [{"kind": "performance"}],
                "model": {"path": "model.tflite"},
            },
            {
                "results": [{"kind": "compatibility"}],
                "target": {"name": "ethos-u55"},
            },
        ]

        merged = (
            reporter._merge_standardized_outputs(  # pylint: disable=protected-access
                outputs
            )
        )

        assert merged["schema_version"] == "1.0.0"
        assert merged["run_id"] == "abc123"
        assert len(merged["results"]) == 2
        assert merged["model"] == {"path": "model.tflite"}
        assert merged["target"] == {"name": "ethos-u55"}
