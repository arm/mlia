# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA advice generation."""

from functools import singledispatchmethod

from mlia.core.advice_generation import FactBasedAdviceProducer, advice_category
from mlia.core.common import AdviceCategory, DataItem
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.target.common.reporters import (
    ModelIsNotTFLiteCompatible,
    TFLiteCompatibilityCheckFailed,
    handle_model_is_not_tflite_compatible_common,
    handle_tflite_check_failed_common,
)
from mlia.target.tosa.data_analysis import (
    ModelIsNotTOSACompatible,
    ModelIsTOSACompatible,
)


class TOSAAdviceProducer(FactBasedAdviceProducer):
    """TOSA advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:  # type: ignore
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_tosa_compatible(
        self, _data_item: ModelIsTOSACompatible
    ) -> None:
        """Advice for TOSA compatibility."""
        self.add_advice(
            message="Model is fully TOSA compatible.",
            category=SchemaAdviceCategory.COMPATIBILITY,
            severity=AdviceSeverity.INFO,
        )

    @produce_advice.register
    @advice_category(AdviceCategory.COMPATIBILITY)
    def handle_model_is_not_tosa_compatible(
        self, _data_item: ModelIsNotTOSACompatible
    ) -> None:
        """Advice for TOSA compatibility."""
        self.add_advice(
            message=(
                "Some operators in the model are not TOSA compatible. "
                "Please, refer to the operators table for more information."
            ),
            category=SchemaAdviceCategory.COMPATIBILITY,
            severity=AdviceSeverity.WARNING,
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
