# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reports module."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from dataclasses import fields
from typing import Any
from typing import Callable

from mlia.backend.vela.compat import Operator
from mlia.backend.vela.compat import Operators
from mlia.backend.vela.performance import layer_metrics
from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import BytesCell
from mlia.core.reporting import Cell
from mlia.core.reporting import ClockCell
from mlia.core.reporting import Column
from mlia.core.reporting import CompoundFormatter
from mlia.core.reporting import CompoundReport
from mlia.core.reporting import CyclesCell
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import SingleRow
from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.common.reporters import report_tflite_compatiblity
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.utils.console import style_improvement
from mlia.utils.types import is_list_of


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
                fmt=Format(str_fmt="{:.0f}%".format),
            ),
        )
    ]

    return SingleRow(
        columns, rows, name="Operators statistics", alias="operators_stats"
    )


def report_operators(ops: list[Operator]) -> Report:
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
                "NPU" if (npu := op.run_on_npu.supported) else "CPU",
                Format(style=style_improvement(npu)),
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
                    (Cell(item, Format(str_fmt=lambda x: f"* {x}")),)
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


def report_target_details(target_config: EthosUConfiguration) -> Report:
    """Return table representation for the target."""
    compiler_config = target_config.resolved_compiler_config

    memory_dict = dict(
        zip(
            ["Sram", "Dram", "OnChipFlash", "OffChipFlash"],
            [
                compiler_config.sram_memory_data,
                compiler_config.dram_memory_data,
                compiler_config.on_chip_flash_memory_data,
                compiler_config.off_chip_flash_memory_data,
            ],
        )
    )

    memory_dict = {
        key: val
        for key, val in memory_dict.items()
        if not list(asdict(val).values()).count(None) == len(list(asdict(val).values()))
    }

    memory_settings = [
        ReportItem(
            "Const mem area",
            "const_mem_area",
            compiler_config.const_mem_area,
        ),
        ReportItem(
            "Arena mem area",
            "arena_mem_area",
            compiler_config.arena_mem_area,
        ),
        ReportItem(
            "Cache mem area",
            "cache_mem_area",
            compiler_config.cache_mem_area,
        ),
    ]

    if compiler_config.arena_cache_size is not None:
        memory_settings.append(
            ReportItem(
                "Arena cache size",
                "arena_cache_size",
                BytesCell(compiler_config.arena_cache_size),
            )
        )
    mem_areas_settings = [
        ReportItem(
            f"{mem_area_name}",
            mem_area_name,
            None,
            nested_items=[
                ReportItem(
                    "Clock scales",
                    "clock_scales",
                    mem_area_settings.clock_scale,
                ),
                ReportItem(
                    "Burst length",
                    "burst_length",
                    BytesCell(mem_area_settings.burst_length),
                ),
                ReportItem(
                    "Read latency",
                    "read_latency",
                    CyclesCell(mem_area_settings.read_latency),
                ),
                ReportItem(
                    "Write latency",
                    "write_latency",
                    CyclesCell(mem_area_settings.write_latency),
                ),
            ],
        )
        for mem_area_name, mem_area_settings in memory_dict.items()
    ]

    system_settings = [
        ReportItem(
            "Accelerator clock",
            "accelerator_clock",
            ClockCell(compiler_config.core_clock),
        ),
        ReportItem(
            "AXI0 port",
            "axi0_port",
            compiler_config.axi0_port,
        ),
        ReportItem(
            "AXI1 port",
            "axi1_port",
            compiler_config.axi1_port,
        ),
        ReportItem(
            "Memory area settings", "memory_area", None, nested_items=mem_areas_settings
        ),
    ]
    return NestedReport(
        "Target information",
        "target",
        [
            ReportItem("Target", alias="target", value=target_config.target),
            ReportItem("MAC", alias="mac", value=target_config.mac),
            ReportItem(
                "Memory mode",
                alias="memory_mode",
                value=compiler_config.memory_mode,
                nested_items=memory_settings,
            ),
            ReportItem(
                "System config",
                alias="system_config",
                value=compiler_config.system_config,
                nested_items=system_settings,
            ),
        ],
    )


def metrics_as_records(
    perf_metrics: list[PerformanceMetrics],
) -> tuple[list[tuple], list[tuple]]:
    """Convert perf metrics object into list of records."""

    def _layerwise_as_metrics(
        perf_metrics: list[PerformanceMetrics],
    ) -> list[tuple]:
        metric_map = defaultdict(list)  # type: dict[str, list]
        format_types = {int: "12,d", str: "", float: "12.2f"}
        rows = []
        for perf_metric in perf_metrics:
            if perf_metric.layerwise_perf_info:
                for layerwise_metric in perf_metric.layerwise_perf_info.layerwise_info:
                    field_names = [
                        field.name
                        for field in fields(layerwise_metric)
                        if field.name != "name"
                    ]
                    duplicate_idx = 1
                    dict_key = getattr(layerwise_metric, "name")
                    while dict_key in metric_map:
                        dict_key = (
                            getattr(layerwise_metric, "name")
                            + " ("
                            + str(duplicate_idx)
                            + ")"
                        )
                        duplicate_idx += 1
                    for field_name in field_names:
                        metric_map[dict_key].append(
                            getattr(layerwise_metric, field_name)
                        )
                rows = [
                    (
                        name,
                        *(
                            Cell(
                                value,
                                Format(
                                    str_fmt=format_types[type(value)]
                                    if type(value) in format_types
                                    else ""
                                ),
                            )
                            for value in values
                        ),
                    )
                    for name, values in metric_map.items()
                ]
        return rows

    def _cycles_as_records(perf_metrics: list[PerformanceMetrics]) -> list[tuple]:
        metric_map = defaultdict(list)
        for metrics in perf_metrics:
            if not metrics.npu_cycles:
                return []
            metric_map["NPU active cycles"].append(metrics.npu_cycles.npu_active_cycles)
            metric_map["NPU idle cycles"].append(metrics.npu_cycles.npu_idle_cycles)
            metric_map["NPU total cycles"].append(metrics.npu_cycles.npu_total_cycles)

        return [
            (name, *(Cell(value, Format(str_fmt="12,d")) for value in values), "cycles")
            for name, values in metric_map.items()
        ]

    def _memory_usage_as_records(perf_metrics: list[PerformanceMetrics]) -> list[tuple]:
        metric_map = defaultdict(list)
        for metrics in perf_metrics:
            if not metrics.memory_usage:
                return []
            metric_map["SRAM used"].append(metrics.memory_usage.sram_memory_area_size)
            metric_map["DRAM used"].append(metrics.memory_usage.dram_memory_area_size)
            metric_map["On-chip flash used"].append(
                metrics.memory_usage.on_chip_flash_memory_area_size
            )
            metric_map["Off-chip flash used"].append(
                metrics.memory_usage.off_chip_flash_memory_area_size
            )

        return [
            (name, *(Cell(value, Format(str_fmt="12.2f")) for value in values), "KiB")
            for name, values in metric_map.items()
            if all(val > 0 for val in values)
        ]

    def _data_beats_as_records(perf_metrics: list[PerformanceMetrics]) -> list[tuple]:
        metric_map = defaultdict(list)
        for metrics in perf_metrics:
            if not metrics.npu_cycles:
                return []
            metric_map["NPU AXI0 RD data beat received"].append(
                metrics.npu_cycles.npu_axi0_rd_data_beat_received
            )
            metric_map["NPU AXI0 WR data beat written"].append(
                metrics.npu_cycles.npu_axi0_wr_data_beat_written
            )
            metric_map["NPU AXI1 RD data beat received"].append(
                metrics.npu_cycles.npu_axi1_rd_data_beat_received
            )

        return [
            (name, *(Cell(value, Format(str_fmt="12,d")) for value in values), "beats")
            for name, values in metric_map.items()
        ]

    return [
        metrics
        for metrics_func in (
            _memory_usage_as_records,
            _cycles_as_records,
            _data_beats_as_records,
        )
        for metrics in metrics_func(perf_metrics)
    ], _layerwise_as_metrics(perf_metrics)


def report_perf_metrics(
    perf_metrics: PerformanceMetrics | list[PerformanceMetrics],
) -> Report:
    """Return comparison table for the performance metrics."""
    if isinstance(perf_metrics, PerformanceMetrics):
        perf_metrics = [perf_metrics]
    rows, layerwise_rows = metrics_as_records(perf_metrics)

    # Create a seperate table for layerwise data
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
                        (
                            diff := 100
                            - (optimized_value.value / original_value.value * 100)
                        ),
                        Format(str_fmt="15.2f", style=style_improvement(diff > 0)),
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
    if layerwise_rows == []:
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
    return CompoundReport(
        [
            Table(
                columns=[
                    Column("Metric", alias="metric", fmt=Format(wrap_width=30)),
                    Column("Value", alias="value", fmt=Format(wrap_width=15)),
                    Column("Unit", alias="unit", fmt=Format(wrap_width=15)),
                ],
                rows=rows,
                name="Performance metrics",
                alias="performance_metrics",
                notes="IMPORTANT: The performance figures above refer to NPU only",
            ),
            Table(
                columns=[
                    Column(name, alias=alias, fmt=Format(wrap_width=30))
                    for alias, _, name in layer_metrics
                ],
                rows=layerwise_rows,
                name="Layer-Wise Metrics",
                alias="layerwise_metrics",
                notes="",
            ),
        ]
    )


def ethos_u_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    report: Callable[[Any], Report] | None = None

    if isinstance(data, PerformanceMetrics) or is_list_of(data, PerformanceMetrics, 2):
        report = report_perf_metrics

    elif is_list_of(data, Advice):
        report = report_advice

    elif is_list_of(data, Operator):
        report = report_operators

    elif isinstance(data, Operators):
        report = report_operators_stat

    elif isinstance(data, EthosUConfiguration):
        report = report_target_details

    elif isinstance(data, (list, tuple)):
        formatters = [ethos_u_formatters(item) for item in data]
        report = CompoundFormatter(formatters)

    elif isinstance(data, TFLiteCompatibilityInfo):
        report = report_tflite_compatiblity

    else:
        raise RuntimeError(f"Unable to find appropriate formatter for {data}.")

    return report
