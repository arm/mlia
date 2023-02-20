# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U data analysis module."""
from __future__ import annotations

from typing import cast

import pytest

from mlia.backend.vela.compat import NpuSupported
from mlia.backend.vela.compat import Operator
from mlia.backend.vela.compat import Operators
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionError
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode
from mlia.target.common.reporters import ModelHasCustomOperators
from mlia.target.common.reporters import ModelIsNotTFLiteCompatible
from mlia.target.common.reporters import TFLiteCompatibilityCheckFailed
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.data_analysis import AllOperatorsSupportedOnNPU
from mlia.target.ethos_u.data_analysis import EthosUDataAnalyzer
from mlia.target.ethos_u.data_analysis import HasCPUOnlyOperators
from mlia.target.ethos_u.data_analysis import HasUnsupportedOnNPUOperators
from mlia.target.ethos_u.data_analysis import OptimizationDiff
from mlia.target.ethos_u.data_analysis import OptimizationResults
from mlia.target.ethos_u.data_analysis import PerfMetricDiff
from mlia.target.ethos_u.performance import MemoryUsage
from mlia.target.ethos_u.performance import NPUCycles
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.target.registry import profile


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
                    cast(EthosUConfiguration, profile("ethos-u55-256")),
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
                            cast(EthosUConfiguration, profile("ethos-u55-256")),
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
                    cast(EthosUConfiguration, profile("ethos-u55-256")),
                    NPUCycles(1, 2, 3, 4, 5, 6),
                    # memory metrics are in kilobytes
                    MemoryUsage(*[i * 1024 for i in range(1, 6)]),  # type: ignore
                ),
                [],
            ),
            [],
        ],
        [
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE),
            [],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR
            ),
            [ModelHasCustomOperators()],
        ],
        [
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.UNKNOWN_ERROR),
            [TFLiteCompatibilityCheckFailed()],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR
            ),
            [ModelIsNotTFLiteCompatible(custom_ops=[], flex_ops=[])],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_errors=[
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        "custom_op1",
                        [],
                    ),
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                        "flex_op1",
                        [],
                    ),
                ],
            ),
            [
                ModelIsNotTFLiteCompatible(
                    custom_ops=["custom_op1"],
                    flex_ops=["flex_op1"],
                )
            ],
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
