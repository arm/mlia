# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data analysis module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.ngp_graph_compiler.config import NGPGraphCompilerConfig
from mlia.backend.ngp_graph_compiler.output_parsing import NGPPerformanceDatabaseParser
from mlia.backend.ngp_graph_compiler.performance import NGPGraphCompilerOutputFiles
from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceMetrics,
)
from mlia.backend.ngp_graph_compiler.statistics import NGPOperatorPerformanceStats
from mlia.target.hydra.data_analysis import HydraDataAnalyzer
from mlia.target.hydra.data_analysis import NGPGraphCompilerModelPerformanceAnalyzed


@pytest.mark.parametrize(
    "analyzed_data",
    (
        NGPGraphCompilerModelPerformanceAnalyzed(
            NGPGraphCompilerPerformanceMetrics(
                backend_config=NGPGraphCompilerConfig("system.ini", "compiler.ini"),
                output_files=NGPGraphCompilerOutputFiles.from_output_dir(
                    Path("DOES_NOT_EXIST"), "TEST"
                ),
                performance_db_parser=NGPPerformanceDatabaseParser(),
                performance_metrics={
                    "0": NGPOperatorPerformanceStats(
                        op_id=[33],
                        op_cycles=15,
                        total_cycles=18,
                        memory={
                            "L1": {
                                "readBytes": 4,
                                "writeBytes": 6,
                                "trafficCycles": 43530,
                            },
                        },
                        utilization=[
                            {"sectionName": "OutputWriter", "hwUtil": 1},
                            {"sectionName": "VectorEngine", "hwUtil": 1},
                        ],
                        operators=["foo"],
                    )
                },
            )
        ),
    ),
)
def test_hydra_data_analyzer(
    analyzed_data: NGPGraphCompilerModelPerformanceAnalyzed,
) -> None:
    """Test Hydra data analyzer."""
    analyzer = HydraDataAnalyzer()
    analyzer.analyze_data(analyzed_data.metrics)
    assert analyzer.get_analyzed_data() == [analyzed_data]
