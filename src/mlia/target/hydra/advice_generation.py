# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.target.hydra.data_analysis import ModelPerformanceAnalysed


class HydraAdviceProducer(FactBasedAdviceProducer):
    """Hydra advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:  # type: ignore
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.PERFORMANCE)
    def handle_model_is_hydra_compatible(
        self, data_item: ModelPerformanceAnalysed
    ) -> None:
        """Advice for Hydra compatibility."""
        self.add_advice(
            [
                "TODO: Model has performance. Argo said so.",
            ]
        )
        raise NotImplementedError(f"TODO: Implement hydra advice using: {data_item=}")
