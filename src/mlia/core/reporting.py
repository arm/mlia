# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reporting module."""
from __future__ import annotations

import json
import logging
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from textwrap import fill
from textwrap import indent
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable

import numpy as np

from mlia.core.typing import OutputFormat
from mlia.utils.console import apply_style
from mlia.utils.console import produce_table
from mlia.utils.types import is_list_of

logger = logging.getLogger(__name__)

OUTPUT_FORMATS = ("json",)


class Report(ABC):
    """Abstract class for the report."""

    @abstractmethod
    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format."""

    @abstractmethod
    def to_plain_text(self, **kwargs: Any) -> str:
        """Convert to human readable format."""


class ReportItem:
    """Item of the report."""

    def __init__(
        self,
        name: str,
        alias: str | None = None,
        value: str | int | Cell | None = None,
        nested_items: list[ReportItem] | None = None,
    ) -> None:
        """Init the report item."""
        self.name = name
        self.alias = alias
        self.value = value
        self.nested_items = nested_items or []

    @property
    def compound(self) -> bool:
        """Return true if item has nested items."""
        return self.nested_items is not None and len(self.nested_items) > 0

    @property
    def raw_value(self) -> Any:
        """Get actual item value."""
        val = self.value
        if isinstance(val, Cell):
            return val.value

        return val


@dataclass
class Format:
    """Column or cell format.

    Format could be applied either to a column or an individual cell.

    :param wrap_width: width of the wrapped text value
    :param str_fmt: string format to be applied to the value
    :param style: text style
    """

    wrap_width: int | None = None
    str_fmt: str | Callable[[Any], str] | None = None
    style: str | None = None


@dataclass
class Cell:
    """Cell definition.

    This a wrapper class for a particular value in the table. Could be used
    for applying specific format to this value.
    """

    value: Any
    fmt: Format | None = None

    def _apply_style(self, value: str) -> str:
        """Apply style to the value."""
        if self.fmt and self.fmt.style:
            value = apply_style(value, self.fmt.style)

        return value

    def _get_value(self) -> str:
        """Return cell value."""
        if self.fmt:
            if isinstance(self.fmt.str_fmt, str):
                return f"{self.value:{self.fmt.str_fmt}}"

            if callable(self.fmt.str_fmt):
                return self.fmt.str_fmt(self.value)

        return str(self.value)

    def __str__(self) -> str:
        """Return string representation."""
        val = self._get_value()
        return self._apply_style(val)

    def to_json(self) -> Any:
        """Cell definition for json."""
        return self.value


class CountAwareCell(Cell):
    """Count aware cell."""

    def __init__(
        self,
        value: int | float | None,
        singular: str,
        plural: str,
        format_string: str = ",d",
    ):
        """Init cell instance."""
        self.unit = singular if value == 1 else plural

        def format_value(val: int | float | None) -> str:
            """Provide string representation for the value."""
            if val is None:
                return ""

            if val == 1:
                return f"1 {singular}"

            return f"{val:{format_string}} {plural}"

        super().__init__(value, Format(str_fmt=format_value))

    def to_json(self) -> Any:
        """Cell definition for json."""
        return {"value": self.value, "unit": self.unit}


class BytesCell(CountAwareCell):
    """Cell that represents memory size."""

    def __init__(self, value: int | None) -> None:
        """Init cell instance."""
        super().__init__(value, "byte", "bytes")


class CyclesCell(CountAwareCell):
    """Cell that represents cycles."""

    def __init__(self, value: int | float | None) -> None:
        """Init cell instance."""
        super().__init__(value, "cycle", "cycles", ",.0f")


class ClockCell(CountAwareCell):
    """Cell that represents clock value."""

    def __init__(self, value: int | float | None) -> None:
        """Init cell instance."""
        super().__init__(value, "Hz", "Hz", ",.0f")


class Column:
    """Column definition."""

    def __init__(
        self,
        header: str,
        alias: str | None = None,
        fmt: Format | None = None,
        only_for: list[str] | None = None,
    ) -> None:
        """Init column definition.

        :param header: column's header
        :param alias: columns's alias, could be used as column's name
        :param fmt: format that will be applied for all column's values
        :param only_for: list of the formats where this column should be
        represented. May be used to differentiate data representation in
        different formats
        """
        self.header = header
        self.alias = alias
        self.fmt = fmt
        self.only_for = only_for

    def supports_format(self, fmt: OutputFormat) -> bool:
        """Return true if column should be shown."""
        return not self.only_for or fmt in self.only_for


class NestedReport(Report):
    """Report with nested items."""

    def __init__(self, name: str, alias: str, items: list[ReportItem]) -> None:
        """Init nested report."""
        self.name = name
        self.alias = alias
        self.items = items

    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format."""
        per_parent: dict[ReportItem | None, dict] = defaultdict(dict)
        result = per_parent[None]

        def collect_as_dicts(
            item: ReportItem,
            parent: ReportItem | None,
            _prev: ReportItem | None,
            _level: int,
        ) -> None:
            """Collect item values as nested dictionaries."""
            parent_dict = per_parent[parent]

            if item.compound:
                item_dict = per_parent[item]
                parent_dict[item.alias] = item_dict
            else:
                out_dis = (
                    item.value.to_json()
                    if isinstance(item.value, Cell)
                    else item.raw_value
                )
                parent_dict[item.alias] = out_dis

        self._traverse(self.items, collect_as_dicts)

        return {self.alias: result}

    def to_plain_text(self, **kwargs: Any) -> str:
        """Convert to human readable format."""
        header = f"{self.name}:\n"
        processed_items = []

        def convert_to_text(
            item: ReportItem,
            _parent: ReportItem | None,
            prev: ReportItem | None,
            level: int,
        ) -> None:
            """Convert item to text representation."""
            if level >= 1 and prev is not None and (item.compound or prev.compound):
                processed_items.append("")

            val = self._item_value(item, level)
            processed_items.append(val)

        self._traverse(self.items, convert_to_text)
        body = "\n".join(processed_items)

        return header + body

    @staticmethod
    def _item_value(
        item: ReportItem, level: int, tab_size: int = 2, column_width: int = 35
    ) -> str:
        """Get report item value."""
        shift = " " * tab_size * level
        if item.value is None:
            return f"{shift}{item.name}:"

        col1 = f"{shift}{item.name}".ljust(column_width)
        col2 = f"{item.value}".rjust(column_width)

        return col1 + col2

    def _traverse(
        self,
        items: list[ReportItem],
        visit_item: Callable[
            [ReportItem, ReportItem | None, ReportItem | None, int], None
        ],
        level: int = 1,
        parent: ReportItem | None = None,
    ) -> None:
        """Traverse through items."""
        prev = None
        for item in items:
            visit_item(item, parent, prev, level)

            self._traverse(item.nested_items, visit_item, level + 1, item)
            prev = item


class Table(Report):
    """Table definition.

    This class could be used for representing tabular data.
    """

    def __init__(
        self,
        columns: list[Column],
        rows: Collection,
        name: str,
        alias: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Init table definition.

        :param columns: list of the table's columns
        :param rows: list of the table's rows
        :param name: name of the table
        :param alias: alias for the table
        """
        self.columns = columns
        self.rows = rows
        self.name = name
        self.alias = alias
        self.notes = notes

    def to_json(self, **kwargs: Any) -> Iterable:
        """Convert table to dict object."""

        def item_to_json(item: Any) -> Any:
            value = item
            if isinstance(item, Cell):
                value = item.value

            if isinstance(value, Table):
                return value.to_json()

            return value

        json_data = [
            {
                col.alias or col.header: item_to_json(item)
                for (item, col) in zip(row, self.columns)
                if col.supports_format("json")
            }
            for row in self.rows
        ]

        if not self.alias:
            return json_data

        return {self.alias: json_data}

    def to_plain_text(self, **kwargs: Any) -> str:
        """Produce report in human readable format."""
        nested = kwargs.get("nested", False)
        show_headers = kwargs.get("show_headers", True)
        show_title = kwargs.get("show_title", True)
        table_style = kwargs.get("table_style", "default")
        space = kwargs.get("space", False)

        headers = (
            [] if (nested or not show_headers) else [c.header for c in self.columns]
        )

        def item_to_plain_text(item: Any, col: Column) -> str:
            """Convert item to text."""
            if isinstance(item, Table):
                return item.to_plain_text(nested=True, **kwargs)

            if is_list_of(item, str):
                as_text = "\n".join(item)
            else:
                as_text = str(item)

            if col.fmt and col.fmt.wrap_width:
                as_text = fill(as_text, col.fmt.wrap_width)

            return as_text

        title = ""
        if show_title and not nested:
            title = f"{self.name}:\n"

        if space in (True, "top"):
            title = "\n" + title

        footer = ""
        if space in (True, "bottom"):
            footer = "\n"
        if self.notes:
            footer = "\n" + self.notes

        formatted_rows = (
            (
                item_to_plain_text(item, col)
                for item, col in zip(row, self.columns)
                if col.supports_format("plain_text")
            )
            for row in self.rows
        )

        if space == "between":
            formatted_table = "\n\n".join(
                produce_table([row], table_style=table_style) for row in formatted_rows
            )
        else:
            formatted_table = produce_table(
                formatted_rows,
                headers=headers,
                table_style="nested" if nested else table_style,
            )

        return title + formatted_table + footer


class SingleRow(Table):
    """Table with a single row."""

    def to_plain_text(self, **kwargs: Any) -> str:
        """Produce report in human readable format."""
        if len(self.rows) != 1:
            raise RuntimeError(f"Table should have only one row, but has {self.rows}.")

        items = "\n".join(
            column.header.ljust(35) + str(item).rjust(25)
            for row in self.rows
            for item, column in zip(row, self.columns)
            if column.supports_format("plain_text")
        )

        return "\n".join([f"{self.name}:", indent(items, "  ")])


class CompoundReport(Report):
    """Compound report.

    This class could be used for producing multiple reports at once.
    """

    def __init__(self, reports: list[Report]) -> None:
        """Init compound report instance."""
        self.reports = reports

    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format.

        Method attempts to create compound dictionary based on provided
        parts.
        """
        result: dict[str, Any] = {}
        for item in self.reports:
            result.update(item.to_json(**kwargs))

        return result

    def to_plain_text(self, **kwargs: Any) -> str:
        """Convert to human readable format."""
        return "\n".join(item.to_plain_text(**kwargs) for item in self.reports)


class CompoundFormatter:
    """Compound data formatter."""

    def __init__(self, formatters: list[Callable]) -> None:
        """Init compound formatter."""
        self.formatters = formatters

    def __call__(self, data: Any) -> Report:
        """Produce report."""
        reports = [formatter(item) for item, formatter in zip(data, self.formatters)]
        return CompoundReport(reports)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder."""

    def default(self, o: Any) -> Any:
        """Support custom types."""
        if isinstance(o, np.integer):
            return int(o)

        if isinstance(o, np.floating):
            return float(o)

        if isinstance(o, Enum) and isinstance(o.value, str):
            return o.value

        return json.JSONEncoder.default(self, o)


class Reporter(ABC):
    """Reporter class."""

    def __init__(
        self,
        formatter_resolver: Callable[[Any], Callable[[Any], Report]],
    ) -> None:
        """Init reporter instance."""
        self.formatter_resolver = formatter_resolver
        self.data: list[tuple[Any, Callable[[Any], Report]]] = []

    @abstractmethod
    def submit(self, data_item: Any, **kwargs: Any) -> None:
        """Submit data for the report."""

    def print_delayed(self) -> None:
        """Print delayed reports."""

    def generate_report(self) -> None:
        """Generate report."""

    @abstractmethod
    def produce_report(
        self, data: Any, formatter: Callable[[Any], Report], **kwargs: Any
    ) -> None:
        """Produce report based on provided data."""


class TextReporter(Reporter):
    """Reporter class."""

    def __init__(
        self,
        formatter_resolver: Callable[[Any], Callable[[Any], Report]],
    ) -> None:
        """Init reporter instance."""
        super().__init__(formatter_resolver)
        self.delayed: list[tuple[Any, Callable[[Any], Report]]] = []
        self.output_format: OutputFormat = "plain_text"

    def submit(self, data_item: Any, delay_print: bool = False, **kwargs: Any) -> None:
        """Submit data for the report."""
        formatter = _apply_format_parameters(
            self.formatter_resolver(data_item), self.output_format, **kwargs
        )
        self.data.append((data_item, formatter))

        if delay_print:
            self.delayed.append((data_item, formatter))
        else:
            self.produce_report(
                data_item,
                self.formatter_resolver(data_item),
                **kwargs,
            )

    def print_delayed(self) -> None:
        """Print delayed reports."""
        if self.delayed:
            data, formatters = zip(*self.delayed)
            self.produce_report(
                data,
                formatter=CompoundFormatter(formatters),
            )
            self.delayed = []

    def produce_report(
        self, data: Any, formatter: Callable[[Any], Report], **kwargs: Any
    ) -> None:
        """Produce report based on provided data."""
        formatted_data = formatter(data)
        logger.info(formatted_data.to_plain_text(**kwargs))


class JSONReporter(Reporter):
    """Reporter class."""

    def __init__(
        self,
        formatter_resolver: Callable[[Any], Callable[[Any], Report]],
    ) -> None:
        """Init reporter instance."""
        super().__init__(formatter_resolver)
        self.output_format: OutputFormat = "json"

    def submit(self, data_item: Any, **kwargs: Any) -> None:
        """Submit data for the report."""
        formatter = _apply_format_parameters(
            self.formatter_resolver(data_item), self.output_format, **kwargs
        )
        self.data.append((data_item, formatter))

    def generate_report(self) -> None:
        """Generate report."""
        if not self.data:
            return

        data, formatters = zip(*self.data)
        self.produce_report(
            data,
            formatter=CompoundFormatter(formatters),
        )

    def produce_report(
        self, data: Any, formatter: Callable[[Any], Report], **kwargs: Any
    ) -> None:
        """Produce report based on provided data."""
        formatted_data = formatter(data)
        print(
            json.dumps(
                formatted_data.to_json(**kwargs), indent=4, cls=CustomJSONEncoder
            ),
        )


def _apply_format_parameters(
    formatter: Callable[[Any], Report], output_format: OutputFormat, **kwargs: Any
) -> Callable[[Any], Report]:
    """Wrap report method."""

    def wrapper(data: Any) -> Report:
        report = formatter(data)
        method_name = f"to_{output_format}"
        method = getattr(report, method_name)
        setattr(report, method_name, partial(method, **kwargs))

        return report

    return wrapper
