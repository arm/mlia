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
