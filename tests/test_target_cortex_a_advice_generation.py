# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for advice generation."""
from __future__ import annotations

import pytest

from mlia.backend.armnn_tflite_delegate.compat import (
    ARMNN_TFLITE_DELEGATE,
)
from mlia.core.advice_generation import Advice
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import ExecutionContext
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.target.common.reporters import ModelHasCustomOperators
from mlia.target.common.reporters import ModelIsNotTFLiteCompatible
from mlia.target.common.reporters import TFLiteCompatibilityCheckFailed
from mlia.target.cortex_a.advice_generation import CortexAAdviceProducer
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.data_analysis import ModelIsCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotCortexACompatible

VERSION = CortexAConfiguration.load_profile("cortex-a").armnn_tflite_delegate_version
BACKEND_INFO = (
    f"{ARMNN_TFLITE_DELEGATE['backend']} " f"{ARMNN_TFLITE_DELEGATE['ops'][VERSION]}"
)


@pytest.mark.parametrize(
    "input_data, advice_category, expected_advice",
    [
        [
            ModelIsNotCortexACompatible(BACKEND_INFO, {"UNSUPPORTED_OP"}, {}),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        f"The following operators are not supported by "
                        f"{BACKEND_INFO} and will fall back to the "
                        f"TensorFlow Lite runtime:\n - UNSUPPORTED_OP"
                    ),
                ),
                Advice(
                    id="1",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        f"Please, refer to the full table of operators above "
                        f"for more information. "
                        f"{CortexAAdviceProducer.cortex_a_disclaimer}"
                    ),
                ),
            ],
        ],
        [
            ModelIsNotCortexACompatible(
                BACKEND_INFO,
                {"UNSUPPORTED_OP"},
                {
                    "CONV_2D": ModelIsNotCortexACompatible.ActivationFunctionSupport(
                        used_unsupported={TFL_ACTIVATION_FUNCTION.SIGN_BIT.name},
                        supported={"RELU"},
                    )
                },
            ),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        f"The following operators are not supported by "
                        f"{BACKEND_INFO} and will fall back to the "
                        f"TensorFlow Lite runtime:\n - UNSUPPORTED_OP"
                    ),
                ),
                Advice(
                    id="1",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        f"The fused activation functions of the following "
                        f"operators are not supported by {BACKEND_INFO}. "
                        f"Please consider using one of the supported "
                        f"activation functions instead:\n - CONV_2D\n"
                        f"   - Used unsupported: {{'SIGN_BIT'}}\n"
                        f"   - Supported: {{'RELU'}}"
                    ),
                ),
                Advice(  # After activation message
                    id="2",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        f"Please, refer to the full table of operators "
                        f"above for more information. "
                        f"{CortexAAdviceProducer.cortex_a_disclaimer}"
                    ),
                ),
            ],
        ],
        [
            ModelIsCortexACompatible(BACKEND_INFO),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        f"Model is fully compatible with {BACKEND_INFO} for "
                        f"Cortex-A. {CortexAAdviceProducer.cortex_a_disclaimer}"
                    ),
                )
            ],
        ],
        [
            ModelIsNotTFLiteCompatible(
                flex_ops=["flex_op1", "flex_op2"],
                custom_ops=["custom_op1", "custom_op2"],
            ),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The following operators are not natively supported "
                        "by TensorFlow Lite: flex_op1, flex_op2. Using "
                        "select TensorFlow operators in TensorFlow Lite "
                        "model requires special initialization of "
                        "TFLiteConverter and TensorFlow Lite run-time. "
                        "Please refer to the TensorFlow documentation for "
                        "more details: "
                        "https://www.tensorflow.org/lite/guide/ops_select "
                        "Note, such models are not supported by the ML "
                        "Inference Advisor."
                    ),
                ),
                Advice(
                    id="1",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The following operators appear to be custom and not "
                        "natively supported by TensorFlow Lite: custom_op1, "
                        "custom_op2. Using custom operators in TensorFlow "
                        "Lite model requires special initialization of "
                        "TFLiteConverter and TensorFlow Lite run-time. "
                        "Please refer to the TensorFlow documentation for "
                        "more details: "
                        "https://www.tensorflow.org/lite/guide/ops_custom "
                        "Note, such models are not supported by the ML "
                        "Inference Advisor."
                    ),
                ),
            ],
        ],
        [
            ModelIsNotTFLiteCompatible(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "Model could not be converted into TensorFlow Lite "
                        "format. Please refer to the table for more details."
                    ),
                ),
            ],
        ],
        [
            ModelHasCustomOperators(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "Models with custom operators require special "
                        "initialization and currently are not supported by "
                        "the ML Inference Advisor."
                    ),
                ),
            ],
        ],
        [
            TFLiteCompatibilityCheckFailed(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "Model could not be converted into TensorFlow Lite "
                        "format. Please refer to the table for more details."
                    ),
                ),
            ],
        ],
    ],
)
def test_cortex_a_advice_producer(
    tmpdir: str,
    input_data: DataItem,
    advice_category: set[AdviceCategory],
    expected_advice: list[Advice],
) -> None:
    """Test Cortex-A advice producer."""
    producer = CortexAAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        output_dir=tmpdir,
    )

    producer.set_context(context)
    producer.produce_advice(input_data)

    advice = producer.get_advice()
    assert isinstance(advice, list)
    assert len(advice) == len(expected_advice)
    for actual, expected in zip(advice, expected_advice):
        assert actual.message == expected.message
