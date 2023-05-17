# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra data analysis module."""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor
from mlia.target.hydra.performance import HydraPerformanceMetrics


class HydraDataAnalyzer(FactExtractor):
    """Hydra data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyse the data."""

    @analyze_data.register
    def analyze_performance(self, data_item: HydraPerformanceMetrics) -> None:
        """Analyse operator compatibility information."""
        self.add_fact(ModelPerformanceAnalysed(data_item))


@dataclass
class ModelPerformanceAnalysed(Fact):
    """Model performance was analyzed."""

    metrics: HydraPerformanceMetrics
