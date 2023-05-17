# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data analysis module."""
from __future__ import annotations

from pathlib import Path

from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.data_analysis import HydraDataAnalyzer
from mlia.target.hydra.data_analysis import ModelPerformanceAnalysed
from mlia.target.hydra.performance import HydraPerformanceMetrics


def test_hydra_data_analyzer() -> None:
    """Test Hydra data analyzer."""
    analyzer = HydraDataAnalyzer()
    metrics = HydraPerformanceMetrics(
        target_config=HydraConfiguration(target="hydra"),
        metrics_file=Path("DOES_NOT_EXIST"),
        operator_performance_data=[],
    )
    analyzer.analyze_data(metrics)
    assert analyzer.get_analyzed_data() == [ModelPerformanceAnalysed(metrics)]
