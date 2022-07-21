# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reporting module."""
import csv
import json
import logging
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from io import TextIOWrapper
from pathlib import Path
from textwrap import fill
from textwrap import indent
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from mlia.core._typing import FileLike
from mlia.core._typing import OutputFormat
from mlia.core._typing import PathOrFileLike
from mlia.utils.console import apply_style
from mlia.utils.console import produce_table
from mlia.utils.logging import LoggerWriter
from mlia.utils.types import is_list_of

logger = logging.getLogger(__name__)


class Report(ABC):
    """Abstract class for the report."""

    @abstractmethod
    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format."""

    @abstractmethod
    def to_csv(self, **kwargs: Any) -> List[Any]:
        """Convert to csv serializible format."""

    @abstractmethod
    def to_plain_text(self, **kwargs: Any) -> str:
        """Convert to human readable format."""


class ReportItem:
    """Item of the report."""

    def __init__(
        self,
        name: str,
        alias: Optional[str] = None,
        value: Optional[Union[str, int, "Cell"]] = None,
        nested_items: Optional[List["ReportItem"]] = None,
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

    wrap_width: Optional[int] = None
    str_fmt: Optional[Union[str, Callable[[Any], str]]] = None
    style: Optional[str] = None


@dataclass
class Cell:
    """Cell definition.

    This a wrapper class for a particular value in the table. Could be used
    for applying specific format to this value.
    """

    value: Any
    fmt: Optional[Format] = None

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

    def to_csv(self) -> Any:
        """Cell definition for csv."""
        return self.value

    def to_json(self) -> Any:
        """Cell definition for json."""
        return self.value


class CountAwareCell(Cell):
    """Count aware cell."""

    def __init__(
        self,
        value: Optional[Union[int, float]],
        singular: str,
        plural: str,
        format_string: str = ",d",
    ):
        """Init cell instance."""
        self.unit = singular if value == 1 else plural

        def format_value(val: Optional[Union[int, float]]) -> str:
            """Provide string representation for the value."""
            if val is None:
                return ""

            if val == 1:
                return f"1 {singular}"

            return f"{val:{format_string}} {plural}"

        super().__init__(value, Format(str_fmt=format_value))

    def to_csv(self) -> Any:
        """Cell definition for csv."""
        return {"value": self.value, "unit": self.unit}

    def to_json(self) -> Any:
        """Cell definition for json."""
        return {"value": self.value, "unit": self.unit}


class BytesCell(CountAwareCell):
    """Cell that represents memory size."""

    def __init__(self, value: Optional[int]) -> None:
        """Init cell instance."""
        super().__init__(value, "byte", "bytes")


class CyclesCell(CountAwareCell):
    """Cell that represents cycles."""

    def __init__(self, value: Optional[Union[int, float]]) -> None:
        """Init cell instance."""
        super().__init__(value, "cycle", "cycles", ",.0f")


class ClockCell(CountAwareCell):
    """Cell that represents clock value."""

    def __init__(self, value: Optional[Union[int, float]]) -> None:
        """Init cell instance."""
        super().__init__(value, "Hz", "Hz", ",.0f")


class Column:
    """Column definition."""

    def __init__(
        self,
        header: str,
        alias: Optional[str] = None,
        fmt: Optional[Format] = None,
        only_for: Optional[List[str]] = None,
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

    def supports_format(self, fmt: str) -> bool:
        """Return true if column should be shown."""
        return not self.only_for or fmt in self.only_for


class NestedReport(Report):
    """Report with nested items."""

    def __init__(self, name: str, alias: str, items: List[ReportItem]) -> None:
        """Init nested report."""
        self.name = name
        self.alias = alias
        self.items = items

    def to_csv(self, **kwargs: Any) -> List[Any]:
        """Convert to csv serializible format."""
        result = {}

        def collect_item_values(
            item: ReportItem,
            _parent: Optional[ReportItem],
            _prev: Optional[ReportItem],
            _level: int,
        ) -> None:
            """Collect item values into a dictionary.."""
            if item.value is None:
                return

            if not isinstance(item.value, Cell):
                result[item.alias] = item.raw_value
                return

            csv_value = item.value.to_csv()
            if isinstance(csv_value, dict):
                csv_value = {
                    f"{item.alias}_{key}": value for key, value in csv_value.items()
                }
            else:
                csv_value = {item.alias: csv_value}

            result.update(csv_value)

        self._traverse(self.items, collect_item_values)

        # make list out of the result dictionary
        # first element - keys of the dictionary as headers
        # second element - list of the dictionary values
        return list(zip(*result.items()))

    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format."""
        per_parent: Dict[Optional[ReportItem], Dict] = defaultdict(dict)
        result = per_parent[None]

        def collect_as_dicts(
            item: ReportItem,
            parent: Optional[ReportItem],
            _prev: Optional[ReportItem],
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
            _parent: Optional[ReportItem],
            prev: Optional[ReportItem],
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
        items: List[ReportItem],
        visit_item: Callable[
            [ReportItem, Optional[ReportItem], Optional[ReportItem], int], None
        ],
        level: int = 1,
        parent: Optional[ReportItem] = None,
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
        columns: List[Column],
        rows: Collection,
        name: str,
        alias: Optional[str] = None,
        notes: Optional[str] = None,
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

    def to_csv(self, **kwargs: Any) -> List[Any]:
        """Convert table to csv format."""
        headers = [[c.header for c in self.columns if c.supports_format("csv")]]

        def item_data(item: Any) -> Any:
            if isinstance(item, Cell):
                return item.value

            if isinstance(item, Table):
                return ";".join(
                    str(item_data(cell)) for row in item.rows for cell in row
                )

            return item

        rows = [
            [
                item_data(item)
                for (item, col) in zip(row, self.columns)
                if col.supports_format("csv")
            ]
            for row in self.rows
        ]

        return headers + rows


class SingleRow(Table):
    """Table with a single row."""

    def to_plain_text(self, **kwargs: Any) -> str:
        """Produce report in human readable format."""
        if len(self.rows) != 1:
            raise Exception("Table should have only one row")

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

    def __init__(self, reports: List[Report]) -> None:
        """Init compound report instance."""
        self.reports = reports

    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format.

        Method attempts to create compound dictionary based on provided
        parts.
        """
        result: Dict[str, Any] = {}
        for item in self.reports:
            result.update(item.to_json(**kwargs))

        return result

    def to_csv(self, **kwargs: Any) -> List[Any]:
        """Convert to csv serializible format.

        CSV format does support only one table. In order to be able to export
        multiply tables they should be merged before that. This method tries to
        do next:

        - if all tables have the same length then just concatenate them
        - if one table has many rows and other just one (two with headers), then
          for each row in table with many rows duplicate values from other tables
        """
        csv_data = [item.to_csv() for item in self.reports]
        lengths = [len(csv_item_data) for csv_item_data in csv_data]

        same_length = len(set(lengths)) == 1
        if same_length:
            # all lists are of the same length, merge them into one
            return [[cell for item in row for cell in item] for row in zip(*csv_data)]

        main_obj_indexes = [i for i, item in enumerate(csv_data) if len(item) > 2]
        one_main_obj = len(main_obj_indexes) == 1

        reference_obj_indexes = [i for i, item in enumerate(csv_data) if len(item) == 2]
        other_only_ref_objs = len(reference_obj_indexes) == len(csv_data) - 1

        if one_main_obj and other_only_ref_objs:
            main_obj = csv_data[main_obj_indexes[0]]
            return [
                item
                + [
                    ref_item
                    for ref_table_index in reference_obj_indexes
                    for ref_item in csv_data[ref_table_index][0 if i == 0 else 1]
                ]
                for i, item in enumerate(main_obj)
            ]

        # write tables one after another if there is no other options
        return [row for item in csv_data for row in item]

    def to_plain_text(self, **kwargs: Any) -> str:
        """Convert to human readable format."""
        return "\n".join(item.to_plain_text(**kwargs) for item in self.reports)


class CompoundFormatter:
    """Compound data formatter."""

    def __init__(self, formatters: List[Callable]) -> None:
        """Init compound formatter."""
        self.formatters = formatters

    def __call__(self, data: Any) -> Report:
        """Produce report."""
        reports = [formatter(item) for item, formatter in zip(data, self.formatters)]
        return CompoundReport(reports)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder."""

    def default(self, o: Any) -> Any:
        """Support numpy types."""
        if isinstance(o, np.integer):
            return int(o)

        if isinstance(o, np.floating):
            return float(o)

        return json.JSONEncoder.default(self, o)


def json_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in json format."""
    json_str = json.dumps(report.to_json(**kwargs), indent=4, cls=CustomJSONEncoder)
    print(json_str, file=output)


def text_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in text format."""
    print(report.to_plain_text(**kwargs), file=output)


def csv_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in csv format."""
    csv_writer = csv.writer(output)
    csv_writer.writerows(report.to_csv(**kwargs))


def produce_report(
    data: Any,
    formatter: Callable[[Any], Report],
    fmt: OutputFormat = "plain_text",
    output: Optional[PathOrFileLike] = None,
    **kwargs: Any,
) -> None:
    """Produce report based on provided data."""
    # check if provided format value is supported
    formats = {"json": json_reporter, "plain_text": text_reporter, "csv": csv_reporter}
    if fmt not in formats:
        raise Exception(f"Unknown format {fmt}")

    if output is None:
        output = cast(TextIOWrapper, LoggerWriter(logger, logging.INFO))

    with ExitStack() as exit_stack:
        if isinstance(output, (str, Path)):
            # open file and add it to the ExitStack context manager
            # in that case it will be automatically closed
            stream = exit_stack.enter_context(open(output, "w", encoding="utf-8"))
        else:
            stream = cast(TextIOWrapper, output)

        # convert data into serializable form
        formatted_data = formatter(data)
        # find handler for the format
        format_handler = formats[fmt]
        # produce report in requested format
        format_handler(formatted_data, stream, **kwargs)


class Reporter:
    """Reporter class."""

    def __init__(
        self,
        formatter_resolver: Callable[[Any], Callable[[Any], Report]],
        output_format: OutputFormat = "plain_text",
        print_as_submitted: bool = True,
    ) -> None:
        """Init reporter instance."""
        self.formatter_resolver = formatter_resolver
        self.output_format = output_format
        self.print_as_submitted = print_as_submitted

        self.data: List[Tuple[Any, Callable[[Any], Report]]] = []
        self.delayed: List[Tuple[Any, Callable[[Any], Report]]] = []

    def submit(self, data_item: Any, delay_print: bool = False, **kwargs: Any) -> None:
        """Submit data for the report."""
        if self.print_as_submitted and not delay_print:
            produce_report(
                data_item,
                self.formatter_resolver(data_item),
                fmt="plain_text",
                **kwargs,
            )

        formatter = _apply_format_parameters(
            self.formatter_resolver(data_item), self.output_format, **kwargs
        )
        self.data.append((data_item, formatter))

        if delay_print:
            self.delayed.append((data_item, formatter))

    def print_delayed(self) -> None:
        """Print delayed reports."""
        if not self.delayed:
            return

        data, formatters = zip(*self.delayed)
        produce_report(
            data,
            formatter=CompoundFormatter(formatters),
            fmt="plain_text",
        )
        self.delayed = []

    def generate_report(self, output: Optional[PathOrFileLike]) -> None:
        """Generate report."""
        already_printed = (
            self.print_as_submitted
            and self.output_format == "plain_text"
            and output is None
        )
        if not self.data or already_printed:
            return

        data, formatters = zip(*self.data)
        produce_report(
            data,
            formatter=CompoundFormatter(formatters),
            fmt=self.output_format,
            output=output,
        )


@contextmanager
def get_reporter(
    output_format: OutputFormat,
    output: Optional[PathOrFileLike],
    formatter_resolver: Callable[[Any], Callable[[Any], Report]],
) -> Generator[Reporter, None, None]:
    """Get reporter and generate report."""
    reporter = Reporter(formatter_resolver, output_format)

    yield reporter

    reporter.generate_report(output)


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


def resolve_output_format(output: Optional[PathOrFileLike]) -> OutputFormat:
    """Resolve output format based on the output name."""
    if isinstance(output, (str, Path)):
        format_from_filename = Path(output).suffix.lstrip(".")

        if format_from_filename in ["json", "csv"]:
            return cast(OutputFormat, format_from_filename)

    return "plain_text"
