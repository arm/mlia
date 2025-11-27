# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the common reporters module."""
from __future__ import annotations

from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionError
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode
from mlia.target.common.reporters import analyze_tflite_compatibility_common
from mlia.target.common.reporters import handle_model_is_not_tflite_compatible_common
from mlia.target.common.reporters import handle_tflite_check_failed_common
from mlia.target.common.reporters import ModelHasCustomOperators
from mlia.target.common.reporters import ModelIsNotTFLiteCompatible
from mlia.target.common.reporters import report_tflite_compatiblity
from mlia.target.common.reporters import TFLiteCompatibilityCheckFailed


@pytest.mark.parametrize(
    "compatibility_info, expected_table_name, expected_table_alias, expected_rows",
    [
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.COMPATIBLE,
                conversion_exception=ValueError("Coversion exception"),
            ),
            "TensorFlow Lite compatibility errors",
            "tflite_compatibility",
            [
                (
                    "TensorFlow Lite compatibility check failed with exception",
                    "Coversion exception",
                )
            ],
        ),
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_errors=[
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        operator="custom_op",
                        location=[],
                    )
                ],
            ),
            "TensorFlow Lite conversion errors",
            "tensorflow_lite_conversion_errors",
            [(1, "custom_op", "", "NEEDS_CUSTOM_OPS", "Conversion error")],
        ),
    ],
)
def test_report_tflite_compatiblity(
    compatibility_info: TFLiteCompatibilityInfo,
    expected_table_name: str,
    expected_table_alias: str,
    expected_rows: str,
) -> None:
    """Test report_tflite_compatiblity function."""
    table = report_tflite_compatiblity(compatibility_info)
    assert isinstance(table, Table)
    assert table.name == expected_table_name
    assert table.alias == expected_table_alias
    assert table.rows == expected_rows


@pytest.mark.parametrize("custom_ops", [["custop_op0", "custom_op1"], None])
@pytest.mark.parametrize("flex_ops", [["flex_op0", "flex_op1"], None])
def test_handle_model_is_not_tflite_compatible_common(
    custom_ops: list[str] | None,
    flex_ops: list[str] | None,
) -> None:
    """Test handle_model_is_not_tflite_compatible_common function."""
    advice_producer = MagicMock()

    handle_model_is_not_tflite_compatible_common(
        advice_producer, ModelIsNotTFLiteCompatible(custom_ops, flex_ops)
    )

    expected_calls = []

    if custom_ops:
        expected_calls.append(
            call(
                [
                    "The following operators appear to be custom and not natively "
                    "supported by TensorFlow Lite: "
                    f"{', '.join(custom_ops)}.",
                    "Using custom operators in TensorFlow Lite model "
                    "requires special initialization of TFLiteConverter and "
                    "TensorFlow Lite run-time.",
                    "Please refer to the TensorFlow documentation for more "
                    "details: https://www.tensorflow.org/lite/guide/ops_custom",
                    "Note, such models are not supported by the "
                    "ML Inference Advisor.",
                ]
            )
        )

    if flex_ops:
        expected_calls.append(
            call(
                [
                    "The following operators are not natively "
                    "supported by TensorFlow Lite: "
                    f"{', '.join(flex_ops)}.",
                    "Using select TensorFlow operators in TensorFlow Lite model "
                    "requires special initialization of TFLiteConverter and "
                    "TensorFlow Lite run-time.",
                    "Please refer to the TensorFlow documentation for more "
                    "details: https://www.tensorflow.org/lite/guide/ops_select",
                    "Note, such models are not supported by "
                    "the ML Inference Advisor.",
                ]
            )
        )

    if expected_calls:
        advice_producer.add_advice.assert_has_calls(expected_calls, any_order=True)

    if not flex_ops and not custom_ops:
        expected_calls = [
            call(
                [
                    "Model could not be converted into TensorFlow Lite format.",
                    "Please refer to the table for more details.",
                ]
            ),
        ]
        advice_producer.add_advice.assert_called_once_with(
            [
                "Model could not be converted into TensorFlow Lite format.",
                "Please refer to the table for more details.",
            ]
        )


def test_handle_tflite_check_failed_common() -> None:
    """Test handle_tflite_check_failed_common function."""
    advice_producer = MagicMock()
    handle_tflite_check_failed_common(advice_producer, TFLiteCompatibilityCheckFailed())

    advice_producer.add_advice.assert_called_once_with(
        [
            "Model could not be converted into TensorFlow Lite format.",
            "Please refer to the table for more details.",
        ]
    )


@pytest.mark.parametrize(
    "compatibility_info, expected_add_fact_arg",
    [
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.COMPATIBLE,
            ),
            None,
        ),
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_exception=ValueError("Coversion exception"),
                conversion_errors=[
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        operator="custom_op",
                        location=[],
                    )
                ],
            ),
            ModelIsNotTFLiteCompatible(["custom_op"], []),
        ),
        (
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.UNKNOWN_ERROR),
            TFLiteCompatibilityCheckFailed(),
        ),
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR
            ),
            ModelHasCustomOperators(),
        ),
    ],
)
def test_analyze_tflite_compatibility_common(
    compatibility_info: TFLiteCompatibilityInfo,
    expected_add_fact_arg: Any,
) -> None:
    """Test analyze_tflite_compatibility_common function."""
    advice_producer = MagicMock()
    analyze_tflite_compatibility_common(advice_producer, compatibility_info)

    if expected_add_fact_arg is not None:
        advice_producer.add_fact.assert_called_with(expected_add_fact_arg)
    else:
        advice_producer.add_fact_assert_not_called()
