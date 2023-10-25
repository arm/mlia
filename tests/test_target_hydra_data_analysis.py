# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data analysis module."""
from __future__ import annotations

from pathlib import Path

from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.performance import ArgoPerformanceMetrics
from mlia.target.hydra.data_analysis import HydraDataAnalyzer
from mlia.target.hydra.data_analysis import ModelPerformanceAnalysed


def test_hydra_data_analyzer() -> None:
    """Test Hydra data analyzer."""
    analyzer = HydraDataAnalyzer()
    metrics = ArgoPerformanceMetrics(
        backend_config=ArgoConfig(),
        metrics_file=Path("DOES_NOT_EXIST"),
        operator_performance_data=[],
    )
    analyzer.analyze_data(metrics)
    assert analyzer.get_analyzed_data() == [ModelPerformanceAnalysed(metrics)]
