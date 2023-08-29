# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable

from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import Table
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.performance import HydraPerformanceMetrics
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


def report_hydra_performance(metrics: HydraPerformanceMetrics) -> Report:
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


def hydra_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, HydraConfiguration):
        return report_target

    if isinstance(data, HydraPerformanceMetrics):
        return report_hydra_performance

    raise RuntimeError(f"Unable to find appropriate formatter for {data}.")
