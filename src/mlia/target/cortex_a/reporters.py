# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import cast

from mlia.core.advice_generation import Advice
from mlia.core.reporters import report_advice
from mlia.core.reporting import Cell
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.common.reporters import report_tflite_compatiblity
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.utils.console import style_improvement
from mlia.utils.types import is_list_of


def report_target(target_config: CortexAConfiguration) -> Report:
    """Generate report for the target."""
    return NestedReport(
        "Target information",
        "target",
        [
            ReportItem("Target", alias="target", value=target_config.target),
        ],
    )


def report_cortex_a_operators(op_compat: CortexACompatibilityInfo) -> Report:
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
                "Arm NN TFLite Delegate compatibility",
                alias="cortex_a_compatible",
                fmt=Format(wrap_width=40),
            ),
        ],
        [
            (
                index + 1,
                op.location,
                op.full_name,
                Cell(
                    op_compat.get_support_type(op),
                    Format(
                        wrap_width=30,
                        style=style_improvement(op_compat.is_op_compatible(op)),
                        str_fmt=lambda v: cast(str, v.value),
                    ),
                ),
            )
            for index, op in enumerate(op_compat.operators)
        ],
        name="Operators",
        alias="operators",
    )


def cortex_a_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, CortexAConfiguration):
        return report_target

    if isinstance(data, TFLiteCompatibilityInfo):
        return report_tflite_compatiblity  # type: ignore

    if isinstance(data, CortexACompatibilityInfo):
        return report_cortex_a_operators

    raise Exception(f"Unable to find appropriate formatter for {data}.")
