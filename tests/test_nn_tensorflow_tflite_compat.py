# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tflite_compat module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107
from tensorflow.lite.python import convert

from mlia.nn.tensorflow.tflite_compat import converter_error_data_pb2
from mlia.nn.tensorflow.tflite_compat import TFLiteChecker
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionError
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode


def test_not_fully_compatible_model_flex_ops() -> None:
    """Test models that requires TF_SELECT_OPS."""
    model = keras.models.Sequential(
        [
            keras.layers.Dense(units=1, input_shape=[1], batch_size=1),
            keras.layers.Dense(units=16, activation="softsign"),
            keras.layers.Dense(units=1),
        ]
    )

    checker = TFLiteChecker()
    result = checker.check_compatibility(model)

    assert result.compatible is False
    assert isinstance(result.conversion_exception, convert.ConverterError)
    assert result.conversion_errors is not None
    assert len(result.conversion_errors) == 1

    conv_err = result.conversion_errors[0]
    assert isinstance(conv_err, TFLiteConversionError)
    assert conv_err.message == "'tf.Softsign' op is neither a custom op nor a flex op"
    assert conv_err.code == TFLiteConversionErrorCode.NEEDS_FLEX_OPS
    assert conv_err.operator == "tf.Softsign"
    assert len(conv_err.location) == 3


def _get_tflite_conversion_error(
    error_message: str = "Conversion error",
    custom_op: bool = False,
    flex_op: bool = False,
    unsupported_flow_v1: bool = False,
    gpu_not_compatible: bool = False,
    unknown_reason: bool = False,
) -> convert.ConverterError:
    """Create TensorFlow Lite conversion error."""
    error_data = converter_error_data_pb2.ConverterErrorData
    convert_error = convert.ConverterError(error_message)

    # pylint: disable=no-member
    def _add_error(operator: str, error_code: int) -> None:
        convert_error.append_error(
            error_data(
                operator=error_data.Operator(name=operator),
                error_code=error_code,
                error_message=error_message,
            )
        )

    if custom_op:
        _add_error("custom_op", error_data.ERROR_NEEDS_CUSTOM_OPS)

    if flex_op:
        _add_error("flex_op", error_data.ERROR_NEEDS_FLEX_OPS)

    if unsupported_flow_v1:
        _add_error("flow_op", error_data.ERROR_UNSUPPORTED_CONTROL_FLOW_V1)

    if gpu_not_compatible:
        _add_error("non_gpu_op", error_data.ERROR_GPU_NOT_COMPATIBLE)

    if unknown_reason:
        _add_error("unknown_op", None)  # type: ignore
    # pylint: enable=no-member

    return convert_error


# pylint: disable=undefined-variable,unused-variable
@pytest.mark.parametrize(
    "conversion_error, expected_result",
    [
        (
            None,
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE),
        ),
        (
            err := _get_tflite_conversion_error(custom_op=True),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_exception=err,
                conversion_errors=[
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        operator="custom_op",
                        location=[],
                    )
                ],
            ),
        ),
        (
            err := _get_tflite_conversion_error(flex_op=True),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_exception=err,
                conversion_errors=[
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                        operator="flex_op",
                        location=[],
                    )
                ],
            ),
        ),
        (
            err := _get_tflite_conversion_error(unknown_reason=True),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_exception=err,
                conversion_errors=[
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.UNKNOWN,
                        operator="unknown_op",
                        location=[],
                    )
                ],
            ),
        ),
        (
            err := _get_tflite_conversion_error(
                flex_op=True,
                custom_op=True,
                gpu_not_compatible=True,
                unsupported_flow_v1=True,
            ),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_exception=err,
                conversion_errors=[
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        operator="custom_op",
                        location=[],
                    ),
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                        operator="flex_op",
                        location=[],
                    ),
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.UNSUPPORTED_CONTROL_FLOW_V1,
                        operator="flow_op",
                        location=[],
                    ),
                    TFLiteConversionError(
                        message="Conversion error",
                        code=TFLiteConversionErrorCode.GPU_NOT_COMPATIBLE,
                        operator="non_gpu_op",
                        location=[],
                    ),
                ],
            ),
        ),
        (
            err := _get_tflite_conversion_error(),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_exception=err,
                conversion_errors=[],
            ),
        ),
        (
            err := ValueError("Some unknown issue"),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.UNKNOWN_ERROR,
                conversion_exception=err,
            ),
        ),
        (
            err := ValueError("Unable to restore custom object"),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR,
                conversion_exception=err,
            ),
        ),
        (
            err := FileNotFoundError("Op type not registered"),
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR,
                conversion_exception=err,
            ),
        ),
    ],
)
# pylint: enable=undefined-variable,unused-variable
def test_tflite_compatibility(
    conversion_error: convert.ConverterError | ValueError | None,
    expected_result: TFLiteCompatibilityInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test TensorFlow Lite compatibility."""
    converter_mock = MagicMock()

    if conversion_error is not None:
        converter_mock.convert.side_effect = conversion_error

    monkeypatch.setattr(
        "mlia.nn.tensorflow.tflite_convert.get_tflite_converter",
        lambda *args, **kwargs: converter_mock,
    )

    checker = TFLiteChecker()
    result = checker.check_compatibility(MagicMock())
    assert result == expected_result
