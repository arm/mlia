# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable

from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.data_analysis import ModelPerformanceAnalysed
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


def report_hydra_performance(ops: list[ModelPerformanceAnalysed]) -> Report:
    """Generate report for the operators."""
    raise NotImplementedError("TODO: Implement proper reporting here.")
    # return Table(
    #     [Column("Test.Name"), Column("Test.Value")],
    #     rows=["TEST", "TODO"],
    #     name="hydra_perf",
    # )


def hydra_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, HydraConfiguration):
        return report_target

    if isinstance(data, dict):
        return report_hydra_performance

    raise RuntimeError(f"Unable to find appropriate formatter for {data}.")
