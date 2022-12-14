# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
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
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.target.cortex_a.advice_generation import CortexAAdviceProducer
from mlia.target.cortex_a.data_analysis import ModelHasCustomOperators
from mlia.target.cortex_a.data_analysis import ModelIsCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotTFLiteCompatible
from mlia.target.cortex_a.data_analysis import TFLiteCompatibilityCheckFailed

BACKEND_INFO = (
    f"{ARMNN_TFLITE_DELEGATE['metadata']['backend']} "
    f"{ARMNN_TFLITE_DELEGATE['metadata']['version']}"
)


@pytest.mark.parametrize(
    "input_data, advice_category, expected_advice",
    [
        [
            ModelIsNotCortexACompatible(BACKEND_INFO, {"UNSUPPORTED_OP"}, {}),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "The following operators are not supported by "
                        f"{BACKEND_INFO} and will fall back to the TensorFlow "
                        "Lite runtime:",
                        " - UNSUPPORTED_OP",
                    ]
                ),
                Advice(
                    [
                        "Please, refer to the full table of operators above "
                        "for more information.",
                        CortexAAdviceProducer.cortex_a_disclaimer,
                    ]
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
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "The following operators are not supported by "
                        f"{BACKEND_INFO} and will fall back to the TensorFlow "
                        "Lite runtime:",
                        " - UNSUPPORTED_OP",
                    ]
                ),
                Advice(
                    [
                        "The fused activation functions of the following "
                        f"operators are not supported by {BACKEND_INFO}. "
                        "Please consider using one of the supported activation "
                        "functions instead:",
                        " - CONV_2D\n"
                        "   - Used unsupported: {'SIGN_BIT'}\n"
                        "   - Supported: {'RELU'}",
                    ]
                ),
                Advice(
                    [
                        "Please, refer to the full table of operators above "
                        "for more information.",
                        CortexAAdviceProducer.cortex_a_disclaimer,
                    ]
                ),
            ],
        ],
        [
            ModelIsCortexACompatible(BACKEND_INFO),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        f"Model is fully compatible with {BACKEND_INFO} for Cortex-A.",
                        CortexAAdviceProducer.cortex_a_disclaimer,
                    ]
                )
            ],
        ],
        [
            ModelIsNotTFLiteCompatible(
                flex_ops=["flex_op1", "flex_op2"],
                custom_ops=["custom_op1", "custom_op2"],
            ),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "The following operators are not natively "
                        "supported by TensorFlow Lite: flex_op1, flex_op2.",
                        "Using select TensorFlow operators in TensorFlow Lite model "
                        "requires special initialization of TFLiteConverter and "
                        "TensorFlow Lite run-time.",
                        "Please refer to the TensorFlow documentation for "
                        "more details: "
                        "https://www.tensorflow.org/lite/guide/ops_select",
                        "Note, such models are not supported by "
                        "the ML Inference Advisor.",
                    ]
                ),
                Advice(
                    [
                        "The following operators appear to be custom and not natively "
                        "supported by TensorFlow Lite: custom_op1, custom_op2.",
                        "Using custom operators in TensorFlow Lite model "
                        "requires special initialization of TFLiteConverter and "
                        "TensorFlow Lite run-time.",
                        "Please refer to the TensorFlow documentation for "
                        "more details: "
                        "https://www.tensorflow.org/lite/guide/ops_custom",
                        "Note, such models are not supported by "
                        "the ML Inference Advisor.",
                    ]
                ),
            ],
        ],
        [
            ModelIsNotTFLiteCompatible(),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "Model could not be converted into TensorFlow Lite format.",
                        "Please refer to the table for more details.",
                    ]
                ),
            ],
        ],
        [
            ModelHasCustomOperators(),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "Models with custom operators require special initialization "
                        "and currently are not supported by the ML Inference Advisor.",
                    ]
                ),
            ],
        ],
        [
            TFLiteCompatibilityCheckFailed(),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "Model could not be converted into TensorFlow Lite format.",
                        "Please refer to the table for more details.",
                    ]
                ),
            ],
        ],
    ],
)
def test_cortex_a_advice_producer(
    tmpdir: str,
    input_data: DataItem,
    advice_category: AdviceCategory,
    expected_advice: list[Advice],
) -> None:
    """Test Cortex-A advice producer."""
    producer = CortexAAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        working_dir=tmpdir,
    )

    producer.set_context(context)
    producer.produce_advice(input_data)

    assert producer.get_advice() == expected_advice
