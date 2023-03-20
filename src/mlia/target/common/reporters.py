# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common reports module."""
from __future__ import annotations

from dataclasses import dataclass

from mlia.core.data_analysis import Fact
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import Report
from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo


@dataclass
class ModelIsNotTFLiteCompatible(Fact):
    """Model could not be converted into TensorFlow Lite format."""

    custom_ops: list[str] | None = None
    flex_ops: list[str] | None = None


@dataclass
class TFLiteCompatibilityCheckFailed(Fact):
    """TensorFlow Lite compatibility check failed by unknown reason."""


@dataclass
class ModelHasCustomOperators(Fact):
    """Model could not be loaded because it contains custom ops."""


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


def handle_model_is_not_tflite_compatible_common(  # type: ignore
    self, data_item: ModelIsNotTFLiteCompatible
) -> None:
    """Advice for TensorFlow Lite compatibility."""
    if data_item.flex_ops:
        self.add_advice(
            [
                "The following operators are not natively "
                "supported by TensorFlow Lite: "
                f"{', '.join(data_item.flex_ops)}.",
                "Using select TensorFlow operators in TensorFlow Lite model "
                "requires special initialization of TFLiteConverter and "
                "TensorFlow Lite run-time.",
                "Please refer to the TensorFlow documentation for more "
                "details: https://www.tensorflow.org/lite/guide/ops_select",
                "Note, such models are not supported by the ML Inference Advisor.",
            ]
        )

    if data_item.custom_ops:
        self.add_advice(
            [
                "The following operators appear to be custom and not natively "
                "supported by TensorFlow Lite: "
                f"{', '.join(data_item.custom_ops)}.",
                "Using custom operators in TensorFlow Lite model "
                "requires special initialization of TFLiteConverter and "
                "TensorFlow Lite run-time.",
                "Please refer to the TensorFlow documentation for more "
                "details: https://www.tensorflow.org/lite/guide/ops_custom",
                "Note, such models are not supported by the ML Inference Advisor.",
            ]
        )

    if not data_item.flex_ops and not data_item.custom_ops:
        self.add_advice(
            [
                "Model could not be converted into TensorFlow Lite format.",
                "Please refer to the table for more details.",
            ]
        )


def handle_tflite_check_failed_common(  # type: ignore
    self, _data_item: TFLiteCompatibilityCheckFailed
) -> None:
    """Advice for the failed TensorFlow Lite compatibility checks."""
    self.add_advice(
        [
            "Model could not be converted into TensorFlow Lite format.",
            "Please refer to the table for more details.",
        ]
    )


def analyze_tflite_compatibility_common(  # type: ignore
    self, data_item: TFLiteCompatibilityInfo
) -> None:
    """Analyze TensorFlow Lite compatibility information."""
    if data_item.compatible:
        return

    if data_item.conversion_failed_with_errors:
        self.add_fact(
            ModelIsNotTFLiteCompatible(
                custom_ops=data_item.required_custom_ops,
                flex_ops=data_item.required_flex_ops,
            )
        )

    if data_item.check_failed_with_unknown_error:
        self.add_fact(TFLiteCompatibilityCheckFailed())

    if data_item.conversion_failed_for_model_with_custom_ops:
        self.add_fact(ModelHasCustomOperators())
