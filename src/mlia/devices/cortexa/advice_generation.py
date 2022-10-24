# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.devices.cortexa.data_analysis import ModelIsCortexACompatible
from mlia.devices.cortexa.data_analysis import ModelIsNotCortexACompatible
from mlia.devices.cortexa.data_analysis import ModelIsNotTFLiteCompatible


class CortexAAdviceProducer(FactBasedAdviceProducer):
    """Cortex-A advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.ALL, AdviceCategory.OPERATORS)
    def handle_model_is_cortex_a_compatible(
        self, _data_item: ModelIsCortexACompatible
    ) -> None:
        """Advice for Cortex-A compatibility."""
        self.add_advice(["Model is fully compatible with Cortex-A."])

    @produce_advice.register
    @advice_category(AdviceCategory.ALL, AdviceCategory.OPERATORS)
    def handle_model_is_not_cortex_a_compatible(
        self, _data_item: ModelIsNotCortexACompatible
    ) -> None:
        """Advice for Cortex-A compatibility."""
        self.add_advice(
            [
                "Some operators in the model are not compatible with Cortex-A. "
                "Please, refer to the operators table for more information."
            ]
        )

    @produce_advice.register
    @advice_category(AdviceCategory.ALL, AdviceCategory.OPERATORS)
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
                    "Please refer to the TensorFlow documentation for more details.",
                ]
            )

        if data_item.custom_ops:
            self.add_advice(
                [
                    "The following operators are custom and not natively "
                    "supported by TensorFlow Lite: "
                    f"{', '.join(data_item.custom_ops)}.",
                    "Please refer to the TensorFlow documentation for more details.",
                ]
            )

        if not data_item.flex_ops and not data_item.custom_ops:
            self.add_advice(
                [
                    "Model could not be converted into TensorFlow Lite format.",
                    "Please refer to the table for more details.",
                ]
            )
