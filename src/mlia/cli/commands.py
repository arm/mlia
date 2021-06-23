"""Cli commands module."""
from textwrap import fill

from mlia.advisor import MLAdvisor
from mlia.performance import collect_performance_metrics
from tabulate import tabulate
from typing_extensions import Annotated


class ParamAnnotation:
    """Parameter annotation."""

    def __init__(
        self, description: str = "", required: bool = False, positional: bool = False
    ):
        """Init param annotation."""
        self.description = description
        self.positional = positional
        self.required = required


def operators(
    model: Annotated[str, ParamAnnotation(description="TFLite model", positional=True)]
) -> None:
    """Print the model's operator list."""
    mladvisor = MLAdvisor()
    model_metadata = mladvisor.inspect(model)

    ops = model_metadata.operations()

    table_data = (
        (
            fill(name, 30),
            fill(op_type, 15),
            fill(str(run_on_npu), 20),
            tabulate(
                (
                    (fill(reason, 30), fill(description, 40))
                    for reason, description in reasons
                ),
                tablefmt="plain",
            ),
        )
        for name, op_type, run_on_npu, reasons in ops
    )

    print(
        tabulate(
            table_data,
            headers=["Operation name", "Operation type", "Supported on NPU", "Reason"],
            tablefmt="grid",
        )
    )


def performance(
    model: Annotated[str, ParamAnnotation(description="TFLite model", positional=True)]
) -> None:
    """Print model's performance stats."""
    performance_metrics = collect_performance_metrics(model)

    table_data = (
        (
            "NPU cycles",
            f"{performance_metrics.npu_cycles:12d}",
            performance_metrics.cycles_per_batch_unit,
        ),
        (
            "SRAM Access cycles",
            f"{performance_metrics.sram_access_cycles:12d}",
            performance_metrics.cycles_per_batch_unit,
        ),
        (
            "DRAM Access cycles",
            f"{performance_metrics.dram_access_cycles:12d}",
            performance_metrics.cycles_per_batch_unit,
        ),
        (
            "On-chip Flash Access cycles",
            f"{performance_metrics.on_chip_flash_access_cycles:12d}",
            performance_metrics.cycles_per_batch_unit,
        ),
        (
            "Off-chip Flash Access cycles",
            f"{performance_metrics.off_chip_flash_access_cycles:12d}",
            performance_metrics.cycles_per_batch_unit,
        ),
        (
            "Total cycles",
            f"{performance_metrics.total_cycles:12d}",
            performance_metrics.cycles_per_batch_unit,
        ),
        (
            "Batch Inference cycles",
            f"{performance_metrics.batch_inference_time:7.2f}",
            performance_metrics.inference_time_unit,
        ),
        (
            "Inferences per second",
            f"{performance_metrics.inferences_per_second:7.2f}",
            performance_metrics.inferences_per_second_unit,
        ),
        (
            "Batch size",
            f"{performance_metrics.batch_size:d}",
            "",
        ),
    )

    print(
        tabulate(
            (
                (fill(metric_column, 30), fill(value_column, 15), fill(unit_column, 15))
                for metric_column, value_column, unit_column in table_data
            ),
            headers=["Metric", "Value", "Unit"],
            tablefmt="grid",
        )
    )
