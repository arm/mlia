# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.target.cortex_a.data_analysis import ModelHasCustomOperators
from mlia.target.cortex_a.data_analysis import ModelIsCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotTFLiteCompatible
from mlia.target.cortex_a.data_analysis import TFLiteCompatibilityCheckFailed


class CortexAAdviceProducer(FactBasedAdviceProducer):
    """Cortex-A advice producer."""

    cortex_a_disclaimer = (
        "Note that the provided compatibility information is general. "
        "At runtime individual operators in the given model might fall back to "
        "the TensorFlow Lite reference or might produce errors based on the "
        "specific parameters."
    )

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:  # type: ignore
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_cortex_a_compatible(
        self, data_item: ModelIsCortexACompatible
    ) -> None:
        """Advice for Cortex-A compatibility."""
        self.add_advice(
            [
                f"Model is fully compatible with {data_item.backend_info} for "
                "Cortex-A.",
                self.cortex_a_disclaimer,
            ]
        )

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_not_cortex_a_compatible(
        self, data_item: ModelIsNotCortexACompatible
    ) -> None:
        """Advice for Cortex-A compatibility."""
        if data_item.unsupported_ops:
            self.add_advice(
                [
                    "The following operators are not supported by "
                    f"{data_item.backend_info} and will fall back to the "
                    "TensorFlow Lite runtime:",
                    "\n".join(f" - {op}" for op in data_item.unsupported_ops),
                ]
            )

        if data_item.activation_func_support:
            self.add_advice(
                [
                    "The fused activation functions of the following operators "
                    f"are not supported by {data_item.backend_info}. Please "
                    "consider using one of the supported activation functions "
                    "instead:",
                    "\n".join(
                        f" - {op}\n"
                        f"   - Used unsupported: {act.used_unsupported}\n"
                        f"   - Supported: {act.supported}"
                        for op, act in data_item.activation_func_support.items()
                    ),
                ]
            )

        self.add_advice(
            [
                "Please, refer to the full table of operators above for more "
                "information.",
                self.cortex_a_disclaimer,
            ]
        )

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_not_tflite_compatible(
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

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_tflite_check_failed(
        self, _data_item: TFLiteCompatibilityCheckFailed
    ) -> None:
        """Advice for the failed TensorFlow Lite compatibility checks."""
        self.add_advice(
            [
                "Model could not be converted into TensorFlow Lite format.",
                "Please refer to the table for more details.",
            ]
        )

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_has_custom_operators(
        self, _data_item: ModelHasCustomOperators
    ) -> None:
        """Advice for the models with custom operators."""
        self.add_advice(
            [
                "Models with custom operators require special initialization "
                "and currently are not supported by the ML Inference Advisor.",
            ]
        )
