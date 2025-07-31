# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology advice generation."""
from functools import singledispatchmethod

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.target.neural_technology.data_analysis import (
    NXGraphCompilerModelPerformanceAnalyzed,
)


class NeuralTechnologyAdviceProducer(FactBasedAdviceProducer):
    """Neural Technology advice producer."""

    @singledispatchmethod
    def produce_advice(self, _data_item: DataItem) -> None:  # type: ignore
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.PERFORMANCE)
    def handle_nx_graph_compiler_performance_analyzed(
        self, _: NXGraphCompilerModelPerformanceAnalyzed
    ) -> None:
        """Advice for NT performance estimated by the NX Graph Compiler."""
        self._point_to_performance_table()

    def _point_to_performance_table(self) -> None:
        """Create generic advice for Neural Technology performance."""
        self.add_advice(
            [
                "Please refer to the performance metrics shown in the report",
                "to find possible optimizations.",
            ]
        )
