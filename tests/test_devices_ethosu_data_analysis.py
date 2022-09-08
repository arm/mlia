# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U data analysis module."""
from __future__ import annotations

import pytest

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.data_analysis import AllOperatorsSupportedOnNPU
from mlia.devices.ethosu.data_analysis import EthosUDataAnalyzer
from mlia.devices.ethosu.data_analysis import HasCPUOnlyOperators
from mlia.devices.ethosu.data_analysis import HasUnsupportedOnNPUOperators
from mlia.devices.ethosu.data_analysis import OptimizationDiff
from mlia.devices.ethosu.data_analysis import OptimizationResults
from mlia.devices.ethosu.data_analysis import PerfMetricDiff
from mlia.devices.ethosu.performance import MemoryUsage
from mlia.devices.ethosu.performance import NPUCycles
from mlia.devices.ethosu.performance import OptimizationPerformanceMetrics
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.tools.vela_wrapper import NpuSupported
from mlia.tools.vela_wrapper import Operator
from mlia.tools.vela_wrapper import Operators


def test_perf_metrics_diff() -> None:
    """Test PerfMetricsDiff class."""
    diff_same = PerfMetricDiff(1, 1)
    assert diff_same.same is True
    assert diff_same.improved is False
    assert diff_same.degraded is False
    assert diff_same.diff == 0

    diff_improved = PerfMetricDiff(10, 5)
    assert diff_improved.same is False
    assert diff_improved.improved is True
    assert diff_improved.degraded is False
    assert diff_improved.diff == 50.0

    diff_degraded = PerfMetricDiff(5, 10)
    assert diff_degraded.same is False
    assert diff_degraded.improved is False
    assert diff_degraded.degraded is True
    assert diff_degraded.diff == -100.0

    diff_original_zero = PerfMetricDiff(0, 1)
    assert diff_original_zero.diff == 0


@pytest.mark.parametrize(
    "input_data, expected_facts",
    [
        [
            Operators(
                [
                    Operator(
                        "CPU operator",
                        "CPU operator type",
                        NpuSupported(False, [("CPU only operator", "")]),
                    )
                ]
            ),
            [
                HasCPUOnlyOperators(["CPU operator type"]),
                HasUnsupportedOnNPUOperators(1.0),
            ],
        ],
        [
            Operators(
                [
                    Operator(
                        "NPU operator",
                        "NPU operator type",
                        NpuSupported(True, []),
                    )
                ]
            ),
            [
                AllOperatorsSupportedOnNPU(),
            ],
        ],
        [
            OptimizationPerformanceMetrics(
                PerformanceMetrics(
                    EthosUConfiguration("ethos-u55-256"),
                    NPUCycles(1, 2, 3, 4, 5, 6),
                    # memory metrics are in kilobytes
                    MemoryUsage(*[i * 1024 for i in range(1, 6)]),  # type: ignore
                ),
                [
                    [
                        [
                            OptimizationSettings("pruning", 0.5, None),
                        ],
                        PerformanceMetrics(
                            EthosUConfiguration("ethos-u55-256"),
                            NPUCycles(1, 2, 3, 4, 5, 6),
                            # memory metrics are in kilobytes
                            MemoryUsage(
                                *[i * 1024 for i in range(1, 6)]  # type: ignore
                            ),
                        ),
                    ],
                ],
            ),
            [
                OptimizationResults(
                    [
                        OptimizationDiff(
                            opt_type=[
                                OptimizationSettings("pruning", 0.5, None),
                            ],
                            opt_diffs={
                                "sram": PerfMetricDiff(1.0, 1.0),
                                "dram": PerfMetricDiff(2.0, 2.0),
                                "on_chip_flash": PerfMetricDiff(4.0, 4.0),
                                "off_chip_flash": PerfMetricDiff(5.0, 5.0),
                                "npu_total_cycles": PerfMetricDiff(3, 3),
                            },
                        )
                    ]
                )
            ],
        ],
        [
            OptimizationPerformanceMetrics(
                PerformanceMetrics(
                    EthosUConfiguration("ethos-u55-256"),
                    NPUCycles(1, 2, 3, 4, 5, 6),
                    # memory metrics are in kilobytes
                    MemoryUsage(*[i * 1024 for i in range(1, 6)]),  # type: ignore
                ),
                [],
            ),
            [],
        ],
    ],
)
def test_ethos_u_data_analyzer(
    input_data: DataItem, expected_facts: list[Fact]
) -> None:
    """Test Ethos-U data analyzer."""
    analyzer = EthosUDataAnalyzer()
    analyzer.analyze_data(input_data)
    assert analyzer.get_analyzed_data() == expected_facts
