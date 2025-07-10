# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra data analysis module."""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceMetrics,
)
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor


class HydraDataAnalyzer(FactExtractor):
    """Hydra data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyse the data."""

    @analyze_data.register
    def analyze_ngp_graph_compiler_performance(
        self, data_item: NGPGraphCompilerPerformanceMetrics
    ) -> None:
        """Analyse operator compatibility information."""
        self.add_fact(NGPGraphCompilerModelPerformanceAnalyzed(data_item))


@dataclass
class NGPGraphCompilerModelPerformanceAnalyzed(Fact):
    """Model performance was analyzed with the NGP Graph Compiler."""

    metrics: NGPGraphCompilerPerformanceMetrics
