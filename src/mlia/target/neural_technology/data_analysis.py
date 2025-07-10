# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology data analysis module."""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceMetrics,
)
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor


class NeuralTechnologyDataAnalyzer(FactExtractor):
    """Neural Technology data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyse the data."""

    @analyze_data.register
    def analyze_nx_graph_compiler_performance(
        self, data_item: NXGraphCompilerPerformanceMetrics
    ) -> None:
        """Analyse operator compatibility information."""
        self.add_fact(NXGraphCompilerModelPerformanceAnalyzed(data_item))


@dataclass
class NXGraphCompilerModelPerformanceAnalyzed(Fact):
    """Model performance was analyzed with the Neural Accelerator Graph Compiler."""

    metrics: NXGraphCompilerPerformanceMetrics
