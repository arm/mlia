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

    print("Performance estimation:")

    print(
        "NPU cycles: {} {}".format(
            performance_metrics.npu_cycles, performance_metrics.cycles_per_batch_unit
        )
    )
    print(
        "SRAM Access cycles: {} {}".format(
            performance_metrics.sram_access_cycles,
            performance_metrics.cycles_per_batch_unit,
        )
    )
    print(
        "DRAM Access cycles: {} {}".format(
            performance_metrics.dram_access_cycles,
            performance_metrics.cycles_per_batch_unit,
        )
    )
    print(
        "On-chip Flash Access cycles: {} {}".format(
            performance_metrics.on_chip_flash_access_cycles,
            performance_metrics.cycles_per_batch_unit,
        )
    )
    print(
        "Off-chip Flash Access cycles: {} {}".format(
            performance_metrics.off_chip_flash_access_cycles,
            performance_metrics.cycles_per_batch_unit,
        )
    )
    print(
        "Total cycles: {} {}".format(
            performance_metrics.total_cycles, performance_metrics.cycles_per_batch_unit
        )
    )

    print(
        "Batch Inference time: {} {}".format(
            performance_metrics.batch_inference_time,
            performance_metrics.inference_time_unit,
        )
    )
    print(
        "Inferences per second: {} {}".format(
            performance_metrics.inferences_per_second,
            performance_metrics.inferences_per_second_unit,
        )
    )
    print("Batch size: {}".format(performance_metrics.batch_size))
