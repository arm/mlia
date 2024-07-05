# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable

from mlia.backend.argo.performance import ArgoPerformanceMetrics
from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceMetrics,
)
from mlia.backend.vulkan_model_converter.compat import NGPModelCompatibilityInfo
from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import Cell
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import Table
from mlia.target.hydra.config import HydraConfiguration
from mlia.utils.misc import dict_to_list
from mlia.utils.types import is_list_of


def report_target(target_cfg: HydraConfiguration) -> Report:
    """Generate report for the device."""
    return NestedReport(
        "Target information",
        "target",
        [
            ReportItem("Target", alias="target", value=target_cfg.target),
        ],
    )


def report_hydra_performance(metrics: ArgoPerformanceMetrics) -> Report:
    """Generate report for the operators."""
    op_columns = [
        Column("Operator name", alias="op_name", fmt=Format(wrap_width=25)),
        Column("Type", alias="type", fmt=Format(wrap_width=25)),
    ]
    perf_columns = [
        Column("Pass #", alias="n_pass", fmt=Format(wrap_width=25)),
        Column("HW Block", alias="hw_block", fmt=Format(wrap_width=25)),
        Column(
            "Duration(µs)", alias="duration", fmt=Format(wrap_width=25, str_fmt="12.3f")
        ),
        Column(
            "Percentage of time", alias="percentage_time", fmt=Format(wrap_width=25)
        ),
    ]
    columns = op_columns + perf_columns
    total_time: float = 0
    for _, op_data in enumerate(metrics.operator_performance_data):
        for operator in op_data.get_duration():
            total_time += float(operator[0])

    rows = [
        (
            op_data.name,
            op_data.op_type,
            Table(columns=perf_columns, rows=op_data.get_pass_numbers(), name="Pass #"),
            Table(columns=columns, rows=op_data.get_hw_block(), name="HW Block"),
            Table(
                columns=columns,  # Cell(v, Format(wrap_width=25, str_fmt="12.3f"))
                rows=op_data.get_duration(),
                name="Duration",
            ),
            Table(
                columns=columns,
                rows=op_data.get_percentage_time(total_time),
                name="percentage_time",
            ),
        )
        for i, op_data in enumerate(metrics.operator_performance_data)
    ]

    return Table(
        columns=columns,
        rows=rows,
        name="Argo per-layer analysis",
        alias="argo_per_layer",
    ).sorted_by("duration", True)


def style_improvement(result: bool) -> str:
    """Return different text style based on result."""
    return "green" if result else "yellow"


def report_ngp_compatibility(comp_info: NGPModelCompatibilityInfo) -> Report:
    """Report."""
    return Table(
        [
            Column("#", only_for=["plain_text"]),
            Column(
                "Operator location",
                alias="operator_location",
                fmt=Format(wrap_width=30),
            ),
            Column("Operator type", alias="operator_type", fmt=Format(wrap_width=20)),
            Column(
                "NGP placement",
                alias="ngp_placement",
                fmt=Format(wrap_width=25),
            ),
            Column(
                "NGP compatibility",
                alias="ngp_compatibility",
                fmt=Format(wrap_width=25),
            ),
        ],
        [
            (
                index + 1,
                op.location,
                op.type or "Unknown",
                Cell(
                    op.placement or ("FAIL" if op.error else "Internal Error"),
                    Format(
                        style=style_improvement(op.placement == "NE"),
                    ),
                ),
                op.compat_level or "N/A",
            )
            for index, op in enumerate(comp_info.get_records())
        ],
        name="Operators",
        alias="operators",
    )


def report_ngp_graph_compiler_perf_db(
    metrics: NGPGraphCompilerPerformanceMetrics,
) -> Report:
    """Report NGP graph compiler's graph DB."""
    perf_records = dict(sorted(metrics.performance_metrics.items()))

    general_column = [Column("ID", alias="id", fmt=Format(wrap_width=25))]

    op_columns_titles = {
        "opLocation": "TFLite Operator Location",
        "opType": "TFLite Operator Type",
    }
    op_columns = [
        Column(title, alias=key, fmt=Format(wrap_width=25))
        for key, title in op_columns_titles.items()
    ]

    cycle_columns = [
        Column("Operator Cycles", alias="opCycles", fmt=Format(wrap_width=25)),
        Column("Total Cycles", alias="totalCycles", fmt=Format(wrap_width=25)),
    ]

    hwutil_columns = [
        Column("HW Section", alias="hwSection", fmt=Format(wrap_width=25)),
        Column("HW Utilisation", alias="hwUtil", fmt=Format(wrap_width=25)),
    ]

    mem_column_titles = {
        "memoryName": "Memory Name",
        "readBytes": "Read bytes",
        "writeBytes": "Write bytes",
        "trafficCycles": "Traffic cycles",
    }
    mem_columns = [
        Column(title, alias=key, fmt=Format(wrap_width=25))
        for key, title in mem_column_titles.items()
    ]

    def sub_table_list_values(
        columns: list[Column], keys_values: list[dict[str, Any]], field: str
    ) -> Table:
        return Table(
            columns=columns,
            rows=[[x[field]] for x in keys_values],
            name="Ignored",
        )

    rows = [
        (
            value.op_id,
            *[
                sub_table_list_values(op_columns, value.operators, field)
                for field in op_columns_titles
            ],
            value.op_cycles,
            value.total_cycles,
            sub_table_list_values(hwutil_columns, value.utilization, "sectionName"),
            sub_table_list_values(hwutil_columns, value.utilization, "hwUtil"),
            *[
                sub_table_list_values(
                    mem_columns, dict_to_list(value.memory, "memoryName"), field
                )
                for field in mem_column_titles
            ],
        )
        for _, value in perf_records.items()
    ]

    return Table(
        columns=general_column
        + op_columns
        + cycle_columns
        + hwutil_columns
        + mem_columns,
        rows=rows,
        name="NGP raw performance report",
        alias="ngp_perf_db",
    )


def hydra_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, HydraConfiguration):
        return report_target

    if isinstance(data, ArgoPerformanceMetrics):
        return report_hydra_performance

    if isinstance(data, NGPGraphCompilerPerformanceMetrics):
        return report_ngp_graph_compiler_perf_db

    if isinstance(data, NGPModelCompatibilityInfo):
        return report_ngp_compatibility

    raise RuntimeError(f"Unable to find appropriate formatter for {data}.")
