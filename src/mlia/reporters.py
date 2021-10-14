# Copyright 2021, Arm Ltd.
"""Reports module."""
import csv
import itertools
import json
import logging
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from contextlib import ExitStack
from functools import partial
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
from typing import TextIO
from typing import Tuple
from typing import Union

import pandas as pd
from mlia._typing import FileLike
from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.cli.advice import Advice
from mlia.config import EthosUConfiguration
from mlia.metadata import Operator
from mlia.metadata import Operators
from mlia.metrics import PerformanceMetrics
from mlia.utils.general import is_list_of
from mlia.utils.general import LoggerWriter
from tabulate import tabulate  # type: ignore


LOGGER = logging.getLogger("mlia.reporters")


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


class Format:
    """Column or cell format."""

    def __init__(
        self,
        wrap_width: Optional[int] = None,
        str_fmt: Optional[Union[str, Callable[[Any], str]]] = None,
    ) -> None:
        """Init format instance.

        Format could be applied either to a column or an individual cell.

        :param wrap_width: width of the wrapped text value
        :param str_fmt: string format to be applied to the value
        """
        self.wrap_width = wrap_width
        self.str_fmt = str_fmt


class Cell:
    """Cell definition.

    This a wrapper class for a particular value in the table. Could be used
    for applying specific format to this value.
    """

    def __init__(self, value: Any, fmt: Optional[Format] = None) -> None:
        """Init cell definition.

        :param value: cell's value
        :param fmt: cell's format
        """
        self.value = value
        self.fmt = fmt

    def __str__(self) -> str:
        """Return string representation."""
        if self.fmt:
            if isinstance(self.fmt.str_fmt, str):
                return "{:{fmt}}".format(self.value, fmt=self.fmt.str_fmt)

            if callable(self.fmt.str_fmt):
                return self.fmt.str_fmt(self.value)

        return str(self.value)


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


class ReportDataFrame(Report):
    """Report wrapper for pandas dataframe."""

    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        """Init ReportDataFrame."""
        self.df = df.copy()

    def to_json(self, **kwargs: Any) -> Any:
        """Convert to json serializible format."""
        return self.df.to_dict()

    def to_csv(self, **kwargs: Any) -> List[Any]:
        """Convert to csv serializible format."""
        rows: List = self.df.values.tolist()
        headers: List = self.df.columns.tolist()

        return [headers] + rows

    def to_plain_text(
        self,
        title: Optional[str] = None,
        columns_name: Optional[str] = None,
        showindex: bool = True,
        space: Union[bool, str] = False,
        notes: Optional[str] = None,
        format_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """Convert to human readable format."""
        final_table = ""
        headers = "keys"
        if columns_name:
            headers = [columns_name] + self.df.columns.tolist()

        if title:
            final_table = final_table + title + ":\n"

        if format_mapping:
            if isinstance(format_mapping, dict):
                for field, format_value in format_mapping.items():
                    self.df[field] = self.df[field].apply(format_value.format)

            if callable(format_mapping):
                self.df = self.df.applymap(format_mapping)

        final_table = final_table + tabulate(
            self.df,
            headers=headers,
            tablefmt="fancy_grid",
            numalign="left",
            stralign="left",
            showindex=showindex,
        )
        if space in (True, "top"):
            final_table = "\n" + final_table

        if space in (True, "bottom"):
            final_table = final_table + "\n"

        if notes:
            final_table = final_table + "\n" + notes

        return final_table


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

    def to_plain_text(
        self,
        nested: bool = False,
        show_title: bool = True,
        show_headers: bool = True,
        tablefmt: str = "fancy_grid",
        space: Union[bool, str] = False,
        **kwargs: Any,
    ) -> str:
        """Produce report in human readable format."""
        headers = (
            [] if (nested or not show_headers) else [c.header for c in self.columns]
        )

        def item_to_plain_text(item: Any, col: Column) -> str:
            """Convert item to text."""
            if isinstance(item, Table):
                return item.to_plain_text(True, **kwargs)
            elif is_list_of(item, str):
                as_text = "\n".join(item)
            else:
                as_text = str(item)

            if col.fmt:
                if col.fmt.wrap_width:
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
                tabulate([row], tablefmt=tablefmt, disable_numparse=True)
                for row in formatted_rows
            )
        else:
            formatted_table = tabulate(
                formatted_rows,
                headers=headers,
                tablefmt="plain" if nested else tablefmt,
                disable_numparse=True,
            )

        return title + formatted_table + footer

    def to_csv(self, **kwargs: Any) -> List[Any]:
        """Convert table to csv format."""
        headers = [[c.header for c in self.columns if c.supports_format("csv")]]

        def item_data(item: Any) -> Any:
            if isinstance(item, Cell):
                return item.value

            if isinstance(item, Table):
                return ";".join(str(cell) for row in item.rows for cell in row)

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

    def to_plain_text(
        self,
        nested: bool = False,
        show_title: bool = True,
        show_headers: bool = True,
        tablefmt: str = "fancy_grid",
        space: Union[bool, str] = False,
        **kwargs: Any,
    ) -> str:
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


def report_operators_stat(operators: Operators) -> Report:
    """Return table representation for the ops stats."""
    columns = [
        Column("Number of operators", alias="num_of_operators"),
        Column("Number of NPU supported operators", "num_of_npu_supported_operators"),
        Column("Unsupported ops ratio", "npu_unsupported_ratio"),
    ]
    rows = [
        (
            operators.total_number,
            operators.npu_supported_number,
            Cell(
                operators.npu_unsupported_ratio * 100,
                fmt=Format(str_fmt=lambda x: "{0:.0f}%".format(x)),
            ),
        )
    ]

    return SingleRow(
        columns, rows, name="Operators statistics", alias="operators_stats"
    )


def report_operators(ops: List[Operator]) -> Report:
    """Return table representation for the list of operators."""
    columns = [
        Column("#", only_for=["plain_text"]),
        Column(
            "Operator name",
            alias="operator_name",
            fmt=Format(wrap_width=30),
        ),
        Column(
            "Operator type",
            alias="operator_type",
            fmt=Format(wrap_width=25),
        ),
        Column(
            "Placement",
            alias="placement",
            fmt=Format(wrap_width=20),
        ),
        Column(
            "Notes",
            alias="notes",
            fmt=Format(wrap_width=35),
        ),
    ]

    rows = [
        (
            i + 1,
            op.name,
            op.op_type,
            Cell(
                op.run_on_npu.supported, Format(str_fmt=lambda x: "NPU" if x else "CPU")
            ),
            Table(
                columns=[
                    Column(
                        "Note",
                        alias="note",
                        fmt=Format(wrap_width=35),
                    )
                ],
                rows=[
                    ("* " + item,)
                    for reason in op.run_on_npu.reasons
                    for item in reason
                    if item
                ],
                name="Notes",
            ),
        )
        for i, op in enumerate(ops)
    ]

    return Table(columns, rows, name="Operators", alias="operators")


def report_device(device: EthosUConfiguration) -> Report:
    """Return table representation for the device."""
    columns = [
        Column("IP class", alias="ip_class", fmt=Format(wrap_width=30)),
        Column("MAC", alias="mac"),
        Column("Accelerator config", alias="accelerator_config"),
        Column("System config", alias="system_config"),
        Column("Memory mode", alias="memory_mode"),
    ]

    rows = [
        (
            device.ip_class,
            device.mac,
            device.compiler_options.accelerator_config,
            device.compiler_options.system_config,
            device.compiler_options.memory_mode,
        )
    ]

    return Table(columns, rows, name="Device information", alias="device")


def report_dataframe(df: pd.DataFrame) -> Report:
    """Wrap pandas dataframe into Report type."""
    return ReportDataFrame(df)


def metrics_as_records(perf_metrics: List[PerformanceMetrics]) -> List[Tuple]:
    """Convert perf metrics object into list of records."""
    perf_metrics = [item.in_kilobytes() for item in perf_metrics]

    cycles = (
        (metric, value, "cycles", "12,d")
        for (metric, value,) in [
            ("NPU active cycles", lambda item: item.npu_cycles.npu_active_cycles),
            ("NPU idle cycles", lambda item: item.npu_cycles.npu_idle_cycles),
            ("NPU total cycles", lambda item: item.npu_cycles.npu_total_cycles),
        ]
    )

    data_beats = (
        (metric, value, "beats", "12,d")
        for (metric, value) in [
            (
                "NPU AXI0 RD data beat received",
                lambda item: item.npu_cycles.npu_axi0_rd_data_beat_received,
            ),
            (
                "NPU AXI0 WR data beat written",
                lambda item: item.npu_cycles.npu_axi0_wr_data_beat_written,
            ),
            (
                "NPU AXI1 RD data beat received",
                lambda item: item.npu_cycles.npu_axi1_rd_data_beat_received,
            ),
        ]
    )

    memory_usage = (
        (metric, value, "KiB", "12.2f")
        for (metric, value) in [
            ("SRAM used", lambda item: item.memory_usage.sram_memory_area_size),
            ("DRAM used", lambda item: item.memory_usage.dram_memory_area_size),
            (
                "Unknown memory area used",
                lambda item: item.memory_usage.unknown_memory_area_size,
            ),
            (
                "On-chip flash used",
                lambda item: item.memory_usage.on_chip_flash_memory_area_size,
            ),
            (
                "Off-chip flash used",
                lambda item: item.memory_usage.off_chip_flash_memory_area_size,
            ),
        ]
        if any(value(item) > 0 for item in perf_metrics)  # type: ignore
    )

    return [
        (
            metric,
            *(
                Cell(value(item), Format(str_fmt=fmt))  # type: ignore
                for item in perf_metrics
            ),
            unit,
        )
        for metric, value, unit, fmt in itertools.chain(
            memory_usage, cycles, data_beats
        )
    ]


def report_perf_metrics(
    perf_metrics: Union[PerformanceMetrics, List[PerformanceMetrics]]
) -> Report:
    """Return comprasion table for the performance metrics."""
    if isinstance(perf_metrics, PerformanceMetrics):
        perf_metrics = [perf_metrics]

    rows = metrics_as_records(perf_metrics)

    if len(perf_metrics) == 2:
        return Table(
            columns=[
                Column("Metric", alias="metric", fmt=Format(wrap_width=30)),
                Column("Original", alias="original", fmt=Format(wrap_width=15)),
                Column("Optimized", alias="optimized", fmt=Format(wrap_width=15)),
                Column("Unit", alias="unit", fmt=Format(wrap_width=15)),
                Column("Improvement (%)", alias="improvement"),
            ],
            rows=[
                (
                    metric,
                    original_value,
                    optimized_value,
                    unit,
                    Cell(
                        100 - (optimized_value.value / original_value.value * 100),
                        Format(str_fmt="12.2f"),
                    )
                    if original_value.value != 0
                    else None,
                )
                for metric, original_value, optimized_value, unit in rows
            ],
            name="Performance metrics",
            alias="performance_metrics",
            notes="IMPORTANT: The performance figures above refer to NPU only",
        )

    return Table(
        columns=[
            Column("Metric", alias="metric", fmt=Format(wrap_width=30)),
            Column("Value", alias="value", fmt=Format(wrap_width=15)),
            Column("Unit", alias="unit", fmt=Format(wrap_width=15)),
        ],
        rows=rows,
        name="Performance metrics",
        alias="performance_metrics",
        notes="IMPORTANT: The performance figures above refer to NPU only",
    )


def report_advice(advice: List[Advice]) -> Report:
    """Generate report for the advice."""
    return Table(
        columns=[
            Column("#", only_for=["plain_text"]),
            Column("Advice", alias="advice_message"),
        ],
        rows=[(i + 1, a.advice_msgs) for i, a in enumerate(advice)],
        name="Advice",
        alias="advice",
    )


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


def json_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in json format."""
    json_str = json.dumps(report.to_json(**kwargs), indent=4)
    print(json_str, file=output)


def text_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in text format."""
    print(report.to_plain_text(**kwargs), file=output)


def csv_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in csv format."""
    csv_writer = csv.writer(output)
    csv_writer.writerows(report.to_csv(**kwargs))


def report(
    data: Any,
    formatter: Optional[Callable[[Any], Report]] = None,
    fmt: OutputFormat = "plain_text",
    output: Optional[PathOrFileLike] = None,
    **kwargs: Any,
) -> None:
    """Produce report based on provided data."""
    # check if provided format value is supported
    formats = {"json": json_reporter, "plain_text": text_reporter, "csv": csv_reporter}
    if fmt not in formats:
        raise Exception(f"Unknown format {fmt}")

    if not formatter:
        # if no formatter provided try to find one based on the type of data
        formatter = find_appropriate_formatter(data)

    if output is None:
        output = cast(TextIO, LoggerWriter(LOGGER, logging.INFO))

    with ExitStack() as exit_stack:
        if isinstance(output, (str, Path)):
            # open file and add it to the ExitStack context manager
            # in that case it will be automatically closed
            stream = exit_stack.enter_context(open(output, "w"))
        else:
            stream = output

        # convert data into serializible form
        formatted_data = formatter(data)
        # find handler for the format
        format_handler = formats[fmt]
        # produce report in requested format
        format_handler(formatted_data, stream, **kwargs)


def find_appropriate_formatter(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if isinstance(data, PerformanceMetrics) or is_list_of(data, PerformanceMetrics, 2):
        return report_perf_metrics

    if is_list_of(data, Advice):
        return report_advice

    if is_list_of(data, Operator):
        return report_operators

    if isinstance(data, Operators):
        return report_operators_stat

    if isinstance(data, EthosUConfiguration):
        return report_device

    if isinstance(data, pd.DataFrame):
        return report_dataframe

    if isinstance(data, (list, tuple)):
        formatters = [find_appropriate_formatter(item) for item in data]
        return CompoundFormatter(formatters)

    raise Exception("Unable to find appropriate formatter")


class Reporter:
    """Reporter class."""

    def __init__(
        self,
        output_format: OutputFormat = "plain_text",
        print_as_submitted: bool = True,
    ) -> None:
        """Init reporter instance."""
        self.output_format = output_format
        self.print_as_submitted = print_as_submitted
        self.data: List[Tuple[Any, Callable[[Any], Report]]] = []

    def submit(self, data_item: Any, **kwargs: Any) -> None:
        """Submit data for the report."""
        if self.print_as_submitted:
            report(data_item, fmt="plain_text", **kwargs)

        formatter = _apply_format_parameters(
            find_appropriate_formatter(data_item), self.output_format, **kwargs
        )
        self.data.append((data_item, formatter))

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
        report(
            data,
            formatter=CompoundFormatter(formatters),
            fmt=self.output_format,
            output=output,
        )


@contextmanager
def get_reporter(
    output_format: OutputFormat, output: Optional[PathOrFileLike]
) -> Generator[Reporter, None, None]:
    """Get reporter and generate report."""
    reporter = Reporter(output_format)

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
