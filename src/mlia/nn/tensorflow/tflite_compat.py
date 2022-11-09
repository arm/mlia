# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Functions for checking TensorFlow Lite compatibility."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Any
from typing import cast
from typing import List

import tensorflow as tf
from tensorflow.lite.python import convert

from mlia.nn.tensorflow.utils import get_tflite_converter
from mlia.utils.logging import redirect_raw_output

TF_VERSION_MAJOR, TF_VERSION_MINOR, _ = (int(s) for s in tf.version.VERSION.split("."))
# pylint: disable=import-error,ungrouped-imports
if (TF_VERSION_MAJOR == 2 and TF_VERSION_MINOR > 7) or TF_VERSION_MAJOR > 2:
    from tensorflow.lite.python.metrics import converter_error_data_pb2
else:
    from tensorflow.lite.python.metrics_wrapper import converter_error_data_pb2
# pylint: enable=import-error,ungrouped-imports


logger = logging.getLogger(__name__)


class TFLiteConversionErrorCode(Enum):
    """TensorFlow Lite conversion error codes."""

    NEEDS_FLEX_OPS = auto()
    NEEDS_CUSTOM_OPS = auto()
    UNSUPPORTED_CONTROL_FLOW_V1 = auto()
    GPU_NOT_COMPATIBLE = auto()
    UNKNOWN = auto()


@dataclass
class TFLiteConversionError:
    """TensorFlow Lite conversion error details."""

    message: str
    code: TFLiteConversionErrorCode
    operator: str
    location: list[str]


class TFLiteCompatibilityStatus(Enum):
    """TensorFlow lite compatiblity status."""

    COMPATIBLE = auto()
    TFLITE_CONVERSION_ERROR = auto()
    MODEL_WITH_CUSTOM_OP_ERROR = auto()
    UNKNOWN_ERROR = auto()


@dataclass
class TFLiteCompatibilityInfo:
    """TensorFlow Lite compatibility information."""

    status: TFLiteCompatibilityStatus
    conversion_exception: Exception | None = None
    conversion_errors: list[TFLiteConversionError] | None = None

    def unsupported_ops_by_code(self, code: TFLiteConversionErrorCode) -> list[str]:
        """Filter unsupported operators by error code."""
        if not self.conversion_errors:
            return []

        return [err.operator for err in self.conversion_errors if err.code == code]

    @property
    def compatible(self) -> bool:
        """Return true if model compatible with the TensorFlow Lite format."""
        return self.status == TFLiteCompatibilityStatus.COMPATIBLE

    @property
    def conversion_failed_with_errors(self) -> bool:
        """Return true if conversion to TensorFlow Lite format failed."""
        return self.status == TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR

    @property
    def conversion_failed_for_model_with_custom_ops(self) -> bool:
        """Return true if conversion failed due to custom ops in the model."""
        return self.status == TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR

    @property
    def check_failed_with_unknown_error(self) -> bool:
        """Return true if check failed with unknown error."""
        return self.status == TFLiteCompatibilityStatus.UNKNOWN_ERROR

    @property
    def required_custom_ops(self) -> list[str]:
        """Return list of the custom ops reported during conversion."""
        return self.unsupported_ops_by_code(TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS)

    @property
    def required_flex_ops(self) -> list[str]:
        """Return list of the flex ops reported during conversion."""
        return self.unsupported_ops_by_code(TFLiteConversionErrorCode.NEEDS_FLEX_OPS)


class TFLiteChecker:
    """Class for checking TensorFlow Lite compatibility."""

    def __init__(self, quantized: bool = False) -> None:
        """Init TensorFlow Lite checker."""
        self.quantized = quantized

    def check_compatibility(self, model: Any) -> TFLiteCompatibilityInfo:
        """Check TensorFlow Lite compatibility for the provided model."""
        try:
            logger.debug("Check TensorFlow Lite compatibility for %s", model)
            converter = get_tflite_converter(model, quantized=self.quantized)

            # there is an issue with intercepting TensorFlow output
            # not all output could be captured, for now just intercept
            # stderr output
            with redirect_raw_output(
                logging.getLogger("tensorflow"), stdout_level=None
            ):
                converter.convert()
        except convert.ConverterError as err:
            return self._process_convert_error(err)
        except Exception as err:  # pylint: disable=broad-except
            return self._process_exception(err)

        return TFLiteCompatibilityInfo(
            status=TFLiteCompatibilityStatus.COMPATIBLE,
        )

    def _process_convert_error(
        self, err: convert.ConverterError
    ) -> TFLiteCompatibilityInfo:
        """Parse error details if possible."""
        conversion_errors = None
        if hasattr(err, "errors"):
            conversion_errors = [
                TFLiteConversionError(
                    message=error.error_message.splitlines()[0],
                    code=self._convert_error_code(error.error_code),
                    operator=error.operator.name,
                    location=cast(
                        List[str],
                        [loc.name for loc in error.location.call if loc.name]
                        if hasattr(error, "location")
                        else [],
                    ),
                )
                for error in err.errors
            ]

        return TFLiteCompatibilityInfo(
            status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
            conversion_exception=err,
            conversion_errors=conversion_errors,
        )

    def _process_exception(self, err: Exception) -> TFLiteCompatibilityInfo:
        """Process exception during conversion."""
        status = TFLiteCompatibilityStatus.UNKNOWN_ERROR

        if self._model_with_custom_op(err):
            status = TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR

        return TFLiteCompatibilityInfo(
            status=status,
            conversion_exception=err,
        )

    @staticmethod
    def _model_with_custom_op(err: Exception) -> bool:
        """Check if model could not be loaded because of custom ops."""
        exc_attrs = [
            (
                ValueError,
                [
                    "Unable to restore custom object",
                    "passed to the `custom_objects`",
                ],
            ),
            (
                FileNotFoundError,
                [
                    "Op type not registered",
                ],
            ),
        ]

        return any(
            any(msg in str(err) for msg in messages)
            for exc_type, messages in exc_attrs
            if isinstance(err, exc_type)
        )

    @staticmethod
    def _convert_error_code(code: int) -> TFLiteConversionErrorCode:
        """Convert internal error codes."""
        # pylint: disable=no-member
        error_data = converter_error_data_pb2.ConverterErrorData
        if code == error_data.ERROR_NEEDS_FLEX_OPS:
            return TFLiteConversionErrorCode.NEEDS_FLEX_OPS

        if code == error_data.ERROR_NEEDS_CUSTOM_OPS:
            return TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS

        if code == error_data.ERROR_UNSUPPORTED_CONTROL_FLOW_V1:
            return TFLiteConversionErrorCode.UNSUPPORTED_CONTROL_FLOW_V1

        if code == converter_error_data_pb2.ConverterErrorData.ERROR_GPU_NOT_COMPATIBLE:
            return TFLiteConversionErrorCode.GPU_NOT_COMPATIBLE
        # pylint: enable=no-member

        return TFLiteConversionErrorCode.UNKNOWN
