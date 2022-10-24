# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable

from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import Cell
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import Table
from mlia.devices.tosa.config import TOSAConfiguration
from mlia.devices.tosa.operators import Operator
from mlia.utils.console import style_improvement
from mlia.utils.types import is_list_of


def report_device(device: TOSAConfiguration) -> Report:
    """Generate report for the device."""
    return NestedReport(
        "Device information",
        "device",
        [
            ReportItem("Target", alias="target", value=device.target),
        ],
    )


def report_tosa_operators(ops: list[Operator]) -> Report:
    """Generate report for the operators."""
    return Table(
        [
            Column("#", only_for=["plain_text"]),
            Column(
                "Operator location",
                alias="operator_location",
                fmt=Format(wrap_width=30),
            ),
            Column("Operator name", alias="operator_name", fmt=Format(wrap_width=20)),
            Column(
                "TOSA compatibility",
                alias="is_tosa_compatible",
                fmt=Format(wrap_width=25),
            ),
        ],
        [
            (
                index + 1,
                op.location,
                op.name,
                Cell(
                    op.is_tosa_compatible,
                    Format(
                        style=style_improvement(op.is_tosa_compatible),
                        str_fmt=lambda v: "Compatible" if v else "Not compatible",
                    ),
                ),
            )
            for index, op in enumerate(ops)
        ],
        name="Operators",
        alias="operators",
    )


def tosa_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, TOSAConfiguration):
        return report_device

    if is_list_of(data, Operator):
        return report_tosa_operators

    raise Exception(f"Unable to find appropriate formatter for {data}")
