# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.devices.tosa.data_analysis import ModelIsNotTOSACompatible
from mlia.devices.tosa.data_analysis import ModelIsTOSACompatible


class TOSAAdviceProducer(FactBasedAdviceProducer):
    """TOSA advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.ALL, AdviceCategory.OPERATORS)
    def handle_model_is_tosa_compatible(
        self, _data_item: ModelIsTOSACompatible
    ) -> None:
        """Advice for TOSA compatibility."""
        self.add_advice(["Model is fully TOSA compatible."])

    @produce_advice.register
    @advice_category(AdviceCategory.ALL, AdviceCategory.OPERATORS)
    def handle_model_is_not_tosa_compatible(
        self, _data_item: ModelIsNotTOSACompatible
    ) -> None:
        """Advice for TOSA compatibility."""
        self.add_advice(
            [
                "Some operators in the model are not TOSA compatible. "
                "Please, refer to the operators table for more information."
            ]
        )
