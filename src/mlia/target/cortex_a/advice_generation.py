# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.target.common.reporters import handle_model_is_not_tflite_compatible_common
from mlia.target.common.reporters import handle_tflite_check_failed_common
from mlia.target.common.reporters import ModelHasCustomOperators
from mlia.target.common.reporters import ModelIsNotTFLiteCompatible
from mlia.target.common.reporters import TFLiteCompatibilityCheckFailed
from mlia.target.cortex_a.data_analysis import ModelIsCortexACompatible
from mlia.target.cortex_a.data_analysis import ModelIsNotCortexACompatible


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
            message=(
                f"Model is fully compatible with {data_item.backend_info} for "
                f"Cortex-A. {self.cortex_a_disclaimer}"
            ),
            category=SchemaAdviceCategory.COMPATIBILITY,
            severity=AdviceSeverity.INFO,
        )

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_not_cortex_a_compatible(
        self, data_item: ModelIsNotCortexACompatible
    ) -> None:
        """Advice for Cortex-A compatibility."""
        if data_item.unsupported_ops:
            ops_list = "\n".join(f" - {op}" for op in data_item.unsupported_ops)
            self.add_advice(
                message=(
                    f"The following operators are not supported by "
                    f"{data_item.backend_info} and will fall back to the "
                    f"TensorFlow Lite runtime:\n{ops_list}"
                ),
                category=SchemaAdviceCategory.COMPATIBILITY,
                severity=AdviceSeverity.WARNING,
            )

        if data_item.activation_func_support:
            acts_list = "\n".join(
                f" - {op}\n"
                f"   - Used unsupported: {act.used_unsupported}\n"
                f"   - Supported: {act.supported}"
                for op, act in data_item.activation_func_support.items()
            )
            self.add_advice(
                message=(
                    f"The fused activation functions of the following operators "
                    f"are not supported by {data_item.backend_info}. Please "
                    f"consider using one of the supported activation functions "
                    f"instead:\n{acts_list}"
                ),
                category=SchemaAdviceCategory.COMPATIBILITY,
                severity=AdviceSeverity.WARNING,
            )

        self.add_advice(
            message=(
                "Please, refer to the full table of operators above for more "
                f"information. {self.cortex_a_disclaimer}"
            ),
            category=SchemaAdviceCategory.COMPATIBILITY,
            severity=AdviceSeverity.INFO,
        )

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_not_tflite_compatible(
        self, data_item: ModelIsNotTFLiteCompatible
    ) -> None:
        """Advice for TensorFlow Lite compatibility."""
        handle_model_is_not_tflite_compatible_common(self, data_item)

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_tflite_check_failed(
        self, _data_item: TFLiteCompatibilityCheckFailed
    ) -> None:
        """Advice for the failed TensorFlow Lite compatibility checks."""
        handle_tflite_check_failed_common(self, _data_item)

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_has_custom_operators(
        self, _data_item: ModelHasCustomOperators
    ) -> None:
        """Advice for the models with custom operators."""
        self.add_advice(
            message=(
                "Models with custom operators require special initialization "
                "and currently are not supported by the ML Inference Advisor."
            ),
            category=SchemaAdviceCategory.COMPATIBILITY,
            severity=AdviceSeverity.WARNING,
        )
