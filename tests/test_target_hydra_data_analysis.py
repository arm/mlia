# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data analysis module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.performance import ArgoPerformanceMetrics
from mlia.backend.ngp_graph_compiler.config import NGPGraphCompilerConfig
from mlia.backend.ngp_graph_compiler.output_parsing import NGPPerformanceDatabaseParser
from mlia.backend.ngp_graph_compiler.performance import NGPGraphCompilerOutputFiles
from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceMetrics,
)
from mlia.target.hydra.data_analysis import ArgoModelPerformanceAnalyzed
from mlia.target.hydra.data_analysis import HydraDataAnalyzer
from mlia.target.hydra.data_analysis import NGPGraphCompilerModelPerformanceAnalyzed


@pytest.mark.parametrize(
    "analyzed_data",
    (
        ArgoModelPerformanceAnalyzed(
            ArgoPerformanceMetrics(
                backend_config=ArgoConfig(),
                metrics_file=Path("DOES_NOT_EXIST"),
                operator_performance_data=[],
            )
        ),
        NGPGraphCompilerModelPerformanceAnalyzed(
            NGPGraphCompilerPerformanceMetrics(
                backend_config=NGPGraphCompilerConfig("system.ini", "compiler.ini"),
                output_files=NGPGraphCompilerOutputFiles.from_output_dir(
                    Path("DOES_NOT_EXIST"), "TEST"
                ),
                performance_db_parser=NGPPerformanceDatabaseParser(),
            ),
        ),
    ),
)
def test_hydra_data_analyzer(
    analyzed_data: ArgoModelPerformanceAnalyzed
    | NGPGraphCompilerModelPerformanceAnalyzed,
) -> None:
    """Test Hydra data analyzer."""
    analyzer = HydraDataAnalyzer()
    analyzer.analyze_data(analyzed_data.metrics)
    assert analyzer.get_analyzed_data() == [analyzed_data]
