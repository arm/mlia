# Copyright 2021, Arm Ltd.
"""Reports module."""
from textwrap import fill
from typing import List

from mlia.metadata import Operation
from mlia.metrics import PerformanceMetrics
from tabulate import tabulate


def report_performance_estimation(perf_metrics: PerformanceMetrics) -> None:
    """Produce performance estimation report."""
    table_data = (
        (
            "NPU cycles",
            f"{perf_metrics.npu_cycles:12,d}",
            perf_metrics.cycles_per_batch_unit,
        ),
        (
            "SRAM Access cycles",
            f"{perf_metrics.sram_access_cycles:12,d}",
            perf_metrics.cycles_per_batch_unit,
        ),
        (
            "DRAM Access cycles",
            f"{perf_metrics.dram_access_cycles:12,d}",
            perf_metrics.cycles_per_batch_unit,
        ),
        (
            "On-chip Flash Access cycles",
            f"{perf_metrics.on_chip_flash_access_cycles:12,d}",
            perf_metrics.cycles_per_batch_unit,
        ),
        (
            "Off-chip Flash Access cycles",
            f"{perf_metrics.off_chip_flash_access_cycles:12,d}",
            perf_metrics.cycles_per_batch_unit,
        ),
        (
            "Total cycles",
            f"{perf_metrics.total_cycles:12,d}",
            perf_metrics.cycles_per_batch_unit,
        ),
        (
            "Batch Inference time",
            f"{perf_metrics.batch_inference_time:7,.2f}",
            perf_metrics.inference_time_unit,
        ),
        (
            "Inferences per second",
            f"{perf_metrics.inferences_per_second:7,.2f}",
            perf_metrics.inferences_per_second_unit,
        ),
        (
            "Batch size",
            f"{perf_metrics.batch_size:d}",
            "",
        ),
    )

    print(
        tabulate(
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
    )


def report_supported_operators(ops: List[Operation]) -> None:
    """Produce supported operators report."""
    table_data = (
        (
            fill(op.name, 30),
            fill(op.op_type, 15),
            fill(str(op.run_on_npu.supported), 20),
            tabulate(
                (
                    (fill(reason, 30), fill(description, 40))
                    for reason, description in op.run_on_npu.reasons
                ),
                tablefmt="plain",
            ),
        )
        for op in ops
    )

    print(
        tabulate(
            table_data,
            headers=[
                "Operation name",
                "Operation type",
                "Supported on NPU",
                "Reason",
            ],
            tablefmt="grid",
        )
    )
