# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology data analysis module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.nx_graph_compiler.config import NXGraphCompilerConfig
from mlia.backend.nx_graph_compiler.output_parsing import NXPerformanceDatabaseParser
from mlia.backend.nx_graph_compiler.performance import NXGraphCompilerOutputFiles
from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceMetrics,
)
from mlia.backend.nx_graph_compiler.statistics import NXOperatorPerformanceStats
from mlia.target.neural_technology.data_analysis import NeuralTechnologyDataAnalyzer
from mlia.target.neural_technology.data_analysis import (
    NXGraphCompilerModelPerformanceAnalyzed,
)


@pytest.mark.parametrize(
    "analyzed_data",
    (
        NXGraphCompilerModelPerformanceAnalyzed(
            NXGraphCompilerPerformanceMetrics(
                backend_config=NXGraphCompilerConfig("system.ini", "compiler.ini"),
                output_files=NXGraphCompilerOutputFiles.from_output_dir(
                    Path("DOES_NOT_EXIST"), "TEST"
                ),
                performance_db_parser=NXPerformanceDatabaseParser(),
                performance_metrics={
                    "0": NXOperatorPerformanceStats(
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
def test_neural_technology_data_analyzer(
    analyzed_data: NXGraphCompilerModelPerformanceAnalyzed,
) -> None:
    """Test Neural Technology data analyzer."""
    analyzer = NeuralTechnologyDataAnalyzer()
    analyzer.analyze_data(analyzed_data.metrics)
    assert analyzer.get_analyzed_data() == [analyzed_data]
