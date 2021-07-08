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
from typing import Any
from typing import cast
from typing import Dict
from typing import List

from mlia._typing import FileLike
from mlia._typing import OutputFormat
from mlia._typing import PathOrFileLike
from mlia.metadata import Operation
from mlia.metrics import PerformanceMetrics
from tabulate import tabulate


class DataTypeChecker:
    """Class contains methods for checking data types."""

    def is_perf_metrics(self, data: Any) -> bool:
        """Return true if data is performance metrics."""
        return isinstance(data, PerformanceMetrics)

    def is_operations(self, data: Any) -> bool:
        """Return true if data is list of the model's operations."""
        return isinstance(data, list) and all(
            isinstance(item, Operation) for item in data
        )


class Reporter(ABC):
    """Abstract class for the reporter."""

    def __init__(self) -> None:
        """Init reporter."""
        self.type_checker = DataTypeChecker()

    @abstractmethod
    def supported_formats(self, data: Any) -> List[str]:
        """Abstract method to check supported data."""

    def report(self, data: Any, fmt: str, output: FileLike, **kwargs: Any) -> None:
        """Produce report."""
        if fmt == "txt":
            self.text(data, output, **kwargs)
        elif fmt == "json":
            self.json(data, output, **kwargs)
        elif fmt == "csv":
            self.csv(data, output, **kwargs)
        else:
            raise Exception(f"Unknown format {fmt}")

    def to_json(self, data: Any, **kwargs: Any) -> Dict:
        """Get json representation for the data."""
        raise NotImplementedError()

    def to_csv(self, data: Any, **kwargs: Any) -> List[Any]:
        """Get csv representation for the data."""
        raise NotImplementedError()

    def to_text(self, data: Any, **kwargs: Any) -> str:
        """Get text representation for the data."""
        raise NotImplementedError()

    def json(self, data: Any, output: FileLike, **kwargs: Any) -> None:
        """Produce report in json format."""
        json_data = self.to_json(data, **kwargs)

        json.dump(json_data, output, indent=4)

    def text(self, data: Any, output: FileLike, **kwargs: Any) -> None:
        """Produce report in text format."""
        text_data = self.to_text(data, **kwargs)

        print(text_data, file=output)

    def csv(self, data: Any, output: FileLike, **kwargs: Any) -> None:
        """Produce report in csv format."""
        csv_data = self.to_csv(data, **kwargs)

        csv_writer = csv.writer(output)
        csv_writer.writerows(csv_data)


class PerformanceEstimationReporter(Reporter):
    """Performance estimation reporter."""

    def supported_formats(self, data: Any) -> List[str]:
        """Check if data is performance metrics."""
        if self.type_checker.is_perf_metrics(data):
            return ["txt", "csv", "json"]

        return []

    def to_json(self, data: Any, **kwargs: Any) -> Dict:
        """Get json representation for the data."""
        perf_metrics = cast(PerformanceMetrics, data)
        table_data = self._convert(perf_metrics, False)

        return {
            "performance_metrics": [
                {"metric": metric_column, "value": value_column, "unit": unit_column}
                for metric_column, value_column, unit_column in table_data
            ]
        }

    def to_text(self, data: Any, **kwargs: Any) -> str:
        """Get text representation for the data."""
        perf_metrics = cast(PerformanceMetrics, data)
        table_data = self._convert(perf_metrics)

        return tabulate(
            (
                (
                    fill(metric_column, 30),
                    fill(value_column, 15),
                    fill(unit_column, 15),
                )
                for metric_column, value_column, unit_column in table_data
            ),
            headers=["Metric", "Value", "Unit"],
            tablefmt="grid",
            disable_numparse=True,
        )

    def to_csv(self, data: Any, **kwargs: Any) -> List[Any]:
        """Get csv representation for the data."""
        perf_metrics = cast(PerformanceMetrics, data)

        headers = [("Metric", "Value", "Unit")]
        return headers + self._convert(perf_metrics, False)

    def _convert(
        self, perf_metrics: PerformanceMetrics, convert_to_text: bool = True
    ) -> List[Any]:
        """Convert metrics object to tabular data."""
        cycles = [
            (
                metric,
                f"{value:12,d}" if convert_to_text else value,
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
            (metric, f"{value:7,.2f}" if convert_to_text else value, unit)
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

        batch = [
            (
                "Batch size",
                f"{perf_metrics.batch_size:d}"
                if convert_to_text
                else perf_metrics.batch_size,
                "",
            )
        ]

        return cycles + inferences + batch


class SupportedOperatorsReporter(Reporter):
    """Supporter operators reporter."""

    def supported_formats(self, data: Any) -> List[str]:
        """Check if data is operation info."""
        if self.type_checker.is_operations(data):
            return ["txt", "json", "csv"]

        return []

    def to_json(self, data: Any, **kwargs: Any) -> Dict:
        """Get json representation for the data."""
        ops = cast(List[Operation], data)
        return {
            "operations": [
                {
                    "name": op.name,
                    "type": op.op_type,
                    "npu_supported": op.run_on_npu.supported,
                    "reasons": [
                        {"reason": reason, "description": description}
                        for reason, description in op.run_on_npu.reasons
                    ],
                }
                for op in ops
            ]
        }

    def to_csv(self, data: Any, **kwargs: Any) -> List[Any]:
        """Get csv representation for the data."""
        ops = cast(List[Operation], data)
        headers = [("Name", "Type", "NPU supported")]
        table_data = [(op.name, op.op_type, str(op.run_on_npu.supported)) for op in ops]

        return headers + table_data

    def to_text(self, data: Any, **kwargs: Any) -> str:
        """Get text representation for the data."""
        ops = cast(List[Operation], data)
        table_data = (
            (
                i + 1,
                fill(op.name, 30),
                fill(op.op_type, 25),
                fill("Yes" if op.run_on_npu.supported else "No", 20),
                tabulate(
                    (
                        (fill(reason, 30), fill(description, 40))
                        for reason, description in op.run_on_npu.reasons
                    ),
                    tablefmt="plain",
                ),
            )
            for i, op in enumerate(ops)
        )

        return tabulate(
            table_data,
            headers=[
                "#",
                "Operation name",
                "Operation type",
                "Supported on NPU",
                "Reason",
            ],
            tablefmt="grid",
        )


_reporters: List[Reporter] = []


def register(reporter: Reporter) -> None:
    """Report registration."""
    _reporters.append(reporter)


register(PerformanceEstimationReporter())
register(SupportedOperatorsReporter())


def report(
    data: Any,
    fmt: OutputFormat = "txt",
    output: PathOrFileLike = sys.stdout,
    **kwargs: Any,
) -> None:
    """Produce report based on provided data."""
    for reporter in _reporters:
        if fmt not in reporter.supported_formats(data):
            continue

        with ExitStack() as stack:
            if isinstance(output, (str, Path)):
                stream = stack.enter_context(open(output, "w"))
            else:
                stream = output

            reporter.report(data, fmt, stream, **kwargs)
            break
    else:
        raise Exception(f"No reporter found for {data} and format {fmt}")
