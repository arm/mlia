# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
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
from mlia.devices.cortexa.config import CortexAConfiguration
from mlia.devices.cortexa.operators import Operator
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.utils.console import style_improvement
from mlia.utils.types import is_list_of


def report_device(device: CortexAConfiguration) -> Report:
    """Generate report for the device."""
    return NestedReport(
        "Device information",
        "device",
        [
            ReportItem("Target", alias="target", value=device.target),
        ],
    )


def report_tflite_compatiblity(compat_info: TFLiteCompatibilityInfo) -> Report:
    """Generate report for the TensorFlow Lite compatibility information."""
    if compat_info.conversion_errors:
        return Table(
            [
                Column("#", only_for=["plain_text"]),
                Column("Operator", alias="operator"),
                Column(
                    "Operator location",
                    alias="operator_location",
                    fmt=Format(wrap_width=25),
                ),
                Column("Error code", alias="error_code"),
                Column(
                    "Error message", alias="error_message", fmt=Format(wrap_width=25)
                ),
            ],
            [
                (
                    index + 1,
                    err.operator,
                    ", ".join(err.location),
                    err.code.name,
                    err.message,
                )
                for index, err in enumerate(compat_info.conversion_errors)
            ],
            name="TensorFlow Lite conversion errors",
            alias="tensorflow_lite_conversion_errors",
        )

    return Table(
        columns=[
            Column("Reason", alias="reason"),
            Column(
                "Exception details",
                alias="exception_details",
                fmt=Format(wrap_width=40),
            ),
        ],
        rows=[
            (
                "TensorFlow Lite compatibility check failed with exception",
                str(compat_info.conversion_exception),
            ),
        ],
        name="TensorFlow Lite compatibility errors",
        alias="tflite_compatibility",
    )


def report_cortex_a_operators(ops: list[Operator]) -> Report:
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
                    op.support_type,
                    Format(
                        wrap_width=30,
                        style=style_improvement(op.is_cortex_a_compatible),
                        str_fmt=lambda v: cast(str, v.value),
                    ),
                ),
            )
            for index, op in enumerate(ops)
        ],
        name="Operators",
        alias="operators",
    )


def cortex_a_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, CortexAConfiguration):
        return report_device

    if isinstance(data, TFLiteCompatibilityInfo):
        return report_tflite_compatiblity

    if is_list_of(data, Operator):
        return report_cortex_a_operators

    raise Exception(f"Unable to find appropriate formatter for {data}")
