# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.target.hydra.data_analysis import NGPGraphCompilerModelPerformanceAnalyzed


class HydraAdviceProducer(FactBasedAdviceProducer):
    """Hydra advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:  # type: ignore
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.PERFORMANCE)
    def handle_ngp_graph_compiler_performance_analyzed(
        self, _: NGPGraphCompilerModelPerformanceAnalyzed
    ) -> None:
        """Advice for Hydra performance estimated by the NGP Graph Compiler."""
        self._point_to_performance_table()

    def _point_to_performance_table(self) -> None:
        """Create generic advice for Hydra performance."""
        self.add_advice(
            [
                "Please refer to the performance metrics shown in the report",
                "to find possible optimizations.",
            ]
        )
