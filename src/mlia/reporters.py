# Copyright 2021, Arm Ltd.
"""Reports module."""
import csv
import json
import sys
from abc import ABC
from abc import abstractmethod
from contextlib import ExitStack
from pathlib import Path
from textwrap import fill
from textwrap import indent
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from mlia._typing import FileLike
from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.config import EthosUConfiguration
from mlia.metadata import Operation
from mlia.metadata import Operations
from mlia.metrics import PerformanceMetrics
from tabulate import tabulate


class Report(ABC):
    """Abstract class for the report."""

    @abstractmethod
    def to_json(self) -> Any:
        """Convert to json serializible format."""

    @abstractmethod
    def to_csv(self) -> List:
        """Convert to csv serializible format."""

    @abstractmethod
    def to_text(self) -> str:
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


class Table(Report):
    """Table definition.

    This class could be used for representing tabular data.
    """

    def __init__(
        self,
        columns: List[Column],
        rows: List[Any],
        name: str,
        alias: Optional[str] = None,
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

    def to_json(self) -> Union[Dict, List[Dict]]:
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

    def to_text(self, nested: bool = False) -> str:
        """Produce report in human readable format."""
        headers = [] if nested else [c.header for c in self.columns]

        def item_to_text(item: Any, col: Column) -> str:
            """Convert item to text."""
            if isinstance(item, Table):
                return item.to_text(True)

            as_text = str(item)

            if col.fmt:
                if col.fmt.wrap_width:
                    as_text = fill(as_text, col.fmt.wrap_width)

            return as_text

        return tabulate(
            (
                (
                    item_to_text(item, col)
                    for item, col in zip(row, self.columns)
                    if col.supports_format("txt")
                )
                for row in self.rows
            ),
            headers=headers,
            tablefmt="plain" if nested else "fancy_grid",
            disable_numparse=True,
        )

    def to_csv(self) -> List[Any]:
        """Convert table to csv format."""
        headers = [[c.header for c in self.columns if c.supports_format("csv")]]

        def item_data(item: Any) -> Any:
            if isinstance(item, Cell):
                return item.value

            if isinstance(item, Table):
                return ""

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

    def to_text(self, nested: bool = False) -> str:
        """Produce report in human readable format."""
        if len(self.rows) != 1:
            raise Exception("Table should have only one row")

        items = "\n".join(
            column.header.ljust(35) + str(item).rjust(25)
            for row in self.rows
            for item, column in zip(row, self.columns)
            if column.supports_format("txt")
        )

        return "\n".join([self.name, indent(items, "  ")])


def report_operators_stat(operations: Operations) -> Report:
    """Return table representation for the ops stats."""
    columns = [
        Column("Number of operators", alias="num_of_operators"),
        Column("Number of NPU supported operators", "num_of_npu_supported_operators"),
        Column("Unsupported ops ratio", "npu_unsupported_ratio"),
    ]
    rows = [
        (
            operations.total_number,
            operations.npu_supported_number,
            Cell(
                operations.npu_unsupported_ratio * 100,
                fmt=Format(str_fmt=lambda x: "{0:.0f}%".format(x)),
            ),
        )
    ]

    return SingleRow(
        columns, rows, name="Operators statistics", alias="operators_stats"
    )


def report_operators(ops: List[Operation]) -> Report:
    """Return table representation for the list of operators."""
    columns = [
        Column("#", only_for=["txt"]),
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
            "Supported on NPU",
            alias="supported_on_npu",
            fmt=Format(wrap_width=20),
        ),
        Column("Reason", alias="reason"),
    ]

    rows = [
        (
            i + 1,
            op.name,
            op.op_type,
            Cell(
                op.run_on_npu.supported, Format(str_fmt=lambda x: "Yes" if x else "No")
            ),
            Table(
                columns=[
                    Column(
                        "Reason",
                        alias="reason",
                        fmt=Format(wrap_width=30),
                    ),
                    Column(
                        "Description",
                        alias="description",
                        fmt=Format(wrap_width=40),
                    ),
                ],
                rows=[
                    (reason, description)
                    for reason, description in op.run_on_npu.reasons
                ],
                name="Reasons",
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
        Column("Memory mode", alias="memory_mod"),
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


def report_perf_metrics(perf_metrics: PerformanceMetrics) -> Report:
    """Return table representation for the perf metrics."""
    columns = [
        Column("Metric", alias="metric", fmt=Format(wrap_width=30)),
        Column("Value", alias="value", fmt=Format(wrap_width=15)),
        Column("Unit", alias="unit", fmt=Format(wrap_width=15)),
    ]

    cycles = [
        (
            metric,
            Cell(value, Format(str_fmt="12,d")),
            perf_metrics.cycles_per_batch_unit,
        )
        for (metric, value) in [
            ("NPU cycles", perf_metrics.npu_cycles),
            ("SRAM Access cycles", perf_metrics.sram_access_cycles),
            ("DRAM Access cycles", perf_metrics.dram_access_cycles),
            (
                "On-chip Flash Access cycles",
                perf_metrics.on_chip_flash_access_cycles,
            ),
            (
                "Off-chip Flash Access cycles",
                perf_metrics.off_chip_flash_access_cycles,
            ),
            ("Total cycles", perf_metrics.total_cycles),
        ]
    ]

    inferences = [
        (metric, Cell(value, Format(str_fmt="7,.2f")), unit)
        for metric, value, unit in [
            (
                "Batch Inference time",
                perf_metrics.batch_inference_time,
                perf_metrics.inference_time_unit,
            ),
            (
                "Inferences per second",
                perf_metrics.inferences_per_second,
                perf_metrics.inferences_per_second_unit,
            ),
        ]
    ]

    batch = [("Batch size", Cell(perf_metrics.batch_size, Format(str_fmt="d")), "")]

    memory_usage = [
        (metric, Cell(value / 1024.0, Format(str_fmt="12.2f")), "KiB")
        for (metric, value) in [
            ("Unknown memory area used", perf_metrics.unknown_memory_area_size),
            ("SRAM used", perf_metrics.sram_memory_area_size),
            ("DRAM used", perf_metrics.dram_memory_area_size),
            ("On-chip flash used", perf_metrics.on_chip_flash_memory_area_size),
            ("Off-chip flash used", perf_metrics.off_chip_flash_memory_area_size),
        ]
        if value > 0
    ]

    rows = memory_usage + cycles + inferences + batch

    return Table(columns, rows, name="Overall performance", alias="overall_performance")


class CompoundReport(Report):
    """Compound report.

    This class could be used for producing multiply reports at once.
    """

    def __init__(self, reports: List[Report]) -> None:
        """Init compound report instance."""
        self.reports = reports

    def to_json(self) -> Any:
        """Convert to json serializible format.

        Method attempts to create compound dictionary based on provided
        parts.
        """
        result = {}
        for item in self.reports:
            result.update(item.to_json())

        return result

    def to_csv(self) -> List:
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

        raise Exception("Unable to export data in csv format")

    def to_text(self) -> str:
        """Convert to human readable format."""
        return "\n".join(item.to_text() for item in self.reports)


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
    json.dump(report.to_json(), output, indent=4)


def text_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in text format."""
    print(report.to_text(), file=output)


def csv_reporter(report: Report, output: FileLike, **kwargs: Any) -> None:
    """Produce report in csv format."""
    csv_writer = csv.writer(output)
    csv_writer.writerows(report.to_csv())


def report(
    data: Any,
    formatter: Optional[Callable[[Any], Report]] = None,
    fmt: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **kwargs: Any,
) -> None:
    """Produce report based on provided data."""
    # check if provided format value is supported
    formats = {"json": json_reporter, "txt": text_reporter, "csv": csv_reporter}
    if fmt not in formats:
        raise Exception(f"Unknown format {fmt}")

    if not formatter:
        # if no formatter provided try to find one based on the type of data
        formatter = find_appropriate_formatter(data)

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


def find_appropriate_formatter(data: Any) -> Callable:
    """Find appropriate formatter for the provided data."""
    if isinstance(data, PerformanceMetrics):
        return report_perf_metrics

    if isinstance(data, list) and all(isinstance(item, Operation) for item in data):
        return report_operators

    if isinstance(data, Operations):
        return report_operators_stat

    if isinstance(data, EthosUConfiguration):
        return report_device

    if isinstance(data, (list, tuple)):
        formatters = [find_appropriate_formatter(item) for item in data]
        return CompoundFormatter(formatters)

    raise Exception("Unable to find appropriate formatter")
