# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A data analysis module."""
from __future__ import annotations

import pytest

from mlia.backend.armnn_tflite_delegate.compat import (
    ARMNN_TFLITE_DELEGATE,
)
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionError
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.data_analysis import CortexADataAnalyzer
from mlia.target.cortex_a.data_analysis import ModelHasCustomOperators
from mlia.target.cortex_a.data_analysis import ModelIsCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotTFLiteCompatible
from mlia.target.cortex_a.data_analysis import TFLiteCompatibilityCheckFailed
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.target.cortex_a.operators import Operator

VERSION = CortexAConfiguration.load_profile("cortex-a").armnn_tflite_delegate_version
BACKEND_INFO = f"{ARMNN_TFLITE_DELEGATE['backend']} {VERSION}"


@pytest.mark.parametrize(
    "input_data, expected_facts",
    [
        [
            CortexACompatibilityInfo([], VERSION),
            [ModelIsCortexACompatible(BACKEND_INFO)],
        ],
        [
            CortexACompatibilityInfo(
                [
                    Operator(
                        "CONV_2D",
                        "somewhere",
                        activation_func=TFL_ACTIVATION_FUNCTION.NONE,
                    ),
                    Operator(
                        "CUSTOM",
                        "somewhere else",
                        activation_func=TFL_ACTIVATION_FUNCTION.SIGN_BIT,
                        custom_name="MaxPool3D",
                    ),
                ],
                VERSION,
            ),
            [ModelIsCortexACompatible(BACKEND_INFO)],
        ],
        [
            # pylint: disable=line-too-long
            CortexACompatibilityInfo(
                [
                    Operator(
                        "UNSUPPORTED_OP",
                        "somewhere",
                        activation_func=TFL_ACTIVATION_FUNCTION.NONE,
                    ),
                    Operator(
                        "CUSTOM",
                        "somewhere",
                        activation_func=TFL_ACTIVATION_FUNCTION.NONE,
                        custom_name="UNSUPPORTED_OP",
                    ),
                    Operator(
                        "CONV_2D",
                        "somewhere else",
                        activation_func=TFL_ACTIVATION_FUNCTION.SIGN_BIT,
                    ),
                ],
                VERSION,
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
    analyzed_data = analyzer.get_analyzed_data()
    assert analyzed_data == expected_facts
