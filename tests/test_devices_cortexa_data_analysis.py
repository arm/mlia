# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A data analysis module."""
from __future__ import annotations

import pytest

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.devices.cortexa.data_analysis import CortexADataAnalyzer
from mlia.devices.cortexa.data_analysis import ModelHasCustomOperators
from mlia.devices.cortexa.data_analysis import ModelIsCortexACompatible
from mlia.devices.cortexa.data_analysis import ModelIsNotCortexACompatible
from mlia.devices.cortexa.data_analysis import ModelIsNotTFLiteCompatible
from mlia.devices.cortexa.data_analysis import TFLiteCompatibilityCheckFailed
from mlia.devices.cortexa.operator_compatibility import ARMNN_TFLITE_DELEGATE
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.devices.cortexa.operators import Operator
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionError
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION

BACKEND_INFO = (
    f"{ARMNN_TFLITE_DELEGATE['metadata']['backend']} "
    f"{ARMNN_TFLITE_DELEGATE['metadata']['version']}"
)


@pytest.mark.parametrize(
    "input_data, expected_facts",
    [
        [
            CortexACompatibilityInfo(True, []),
            [ModelIsCortexACompatible(BACKEND_INFO)],
        ],
        [
            CortexACompatibilityInfo(
                True,
                [
                    Operator(
                        "CONV_2D",
                        "somewhere",
                        support_type=Operator.SupportType.COMPATIBLE,
                        activation_func=TFL_ACTIVATION_FUNCTION.NONE,
                    ),
                    Operator(
                        "CUSTOM",
                        "somewhere else",
                        support_type=Operator.SupportType.COMPATIBLE,
                        activation_func=TFL_ACTIVATION_FUNCTION.SIGN_BIT,
                        custom_name="MaxPool3D",
                    ),
                ],
            ),
            [ModelIsCortexACompatible(BACKEND_INFO)],
        ],
        [
            # pylint: disable=line-too-long
            CortexACompatibilityInfo(
                False,
                [
                    Operator(
                        "UNSUPPORTED_OP",
                        "somewhere",
                        support_type=Operator.SupportType.OP_NOT_SUPPORTED,
                        activation_func=TFL_ACTIVATION_FUNCTION.NONE,
                    ),
                    Operator(
                        "CUSTOM",
                        "somewhere",
                        support_type=Operator.SupportType.OP_NOT_SUPPORTED,
                        activation_func=TFL_ACTIVATION_FUNCTION.NONE,
                        custom_name="UNSUPPORTED_OP",
                    ),
                    Operator(
                        "CONV_2D",
                        "somewhere else",
                        support_type=Operator.SupportType.ACTIVATION_NOT_SUPPORTED,
                        activation_func=TFL_ACTIVATION_FUNCTION.SIGN_BIT,
                    ),
                ],
            ),
            [
                ModelIsNotCortexACompatible(
                    BACKEND_INFO,
                    {
                        "UNSUPPORTED_OP",
                        "CUSTOM - 'UNSUPPORTED_OP'",
                    },
                    {
                        "CONV_2D": ModelIsNotCortexACompatible.ActivationFunctionSupport(
                            used_unsupported={TFL_ACTIVATION_FUNCTION.SIGN_BIT.name},
                            supported={
                                "RELU",
                                "RELU6",
                                "RELU_N1_TO_1",
                                "SIGMOID",
                                "TANH",
                                "NONE",
                            },
                        )
                    },
                )
            ],
            # pylint: enable=line-too-long
        ],
        [
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE),
            [],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR
            ),
            [ModelHasCustomOperators()],
        ],
        [
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.UNKNOWN_ERROR),
            [TFLiteCompatibilityCheckFailed()],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR
            ),
            [ModelIsNotTFLiteCompatible(custom_ops=[], flex_ops=[])],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_errors=[
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        "custom_op1",
                        [],
                    ),
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                        "flex_op1",
                        [],
                    ),
                ],
            ),
            [
                ModelIsNotTFLiteCompatible(
                    custom_ops=["custom_op1"],
                    flex_ops=["flex_op1"],
                )
            ],
        ],
    ],
)
def test_cortex_a_data_analyzer(
    input_data: DataItem, expected_facts: list[Fact]
) -> None:
    """Test Cortex-A data analyzer."""
    analyzer = CortexADataAnalyzer()
    analyzer.analyze_data(input_data)
    assert analyzer.get_analyzed_data() == expected_facts
