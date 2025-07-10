# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable

from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceMetrics,
)
from mlia.backend.vulkan_model_converter.compat import NXModelCompatibilityInfo
from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import Cell
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import Table
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.utils.misc import dict_to_list
from mlia.utils.types import is_list_of


def report_target(target_cfg: NeuralTechnologyConfiguration) -> Report:
    """Generate report for the device."""
    return NestedReport(
        "Target information",
        "target",
        [
            ReportItem("Target", alias="target", value=target_cfg.target),
        ],
    )


def style_improvement(result: bool) -> str:
    """Return different text style based on result."""
    return "green" if result else "yellow"


def report_nx_compatibility(comp_info: NXModelCompatibilityInfo) -> Report:
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
                "NX placement",
                alias="nx_placement",
                fmt=Format(wrap_width=25),
            ),
            Column(
                "NX compatibility",
                alias="nx_compatibility",
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


def report_nx_graph_compiler_perf_db(
    metrics: NXGraphCompilerPerformanceMetrics,
) -> Report:
    """Report Neural Accelerator graph compiler's graph DB."""
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
        name="Neural Accelerator raw performance report",
        alias="nx_perf_db",
    )


def neural_technology_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, NeuralTechnologyConfiguration):
        return report_target

    if isinstance(data, NXGraphCompilerPerformanceMetrics):
        return report_nx_graph_compiler_perf_db

    if isinstance(data, NXModelCompatibilityInfo):
        return report_nx_compatibility

    raise RuntimeError(f"Unable to find appropriate formatter for {data}.")
