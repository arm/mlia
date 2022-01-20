# Copyright 2021, Arm Ltd.
"""Tests for Ethos-U advice generation."""
from typing import List

import pytest
from mlia.core.advice_generation import Advice
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.devices.ethosu.advice_generation import EthosUAdviceProducer
from mlia.devices.ethosu.data_analysis import AllOperatorsSupportedOnNPU
from mlia.devices.ethosu.data_analysis import HasCPUOnlyOperators
from mlia.devices.ethosu.data_analysis import HasUnsupportedOnNPUOperators
from mlia.devices.ethosu.data_analysis import OptimizationDiff
from mlia.devices.ethosu.data_analysis import OptimizationResults
from mlia.devices.ethosu.data_analysis import PerfMetricDiff
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings


@pytest.mark.parametrize(
    "input_data, advice_category, expected_advice",
    [
        [
            AllOperatorsSupportedOnNPU(),
            AdviceCategory.OPERATORS_COMPATIBILITY,
            [
                Advice(
                    [
                        "You don't have any unsupported operators, your model will "
                        "run completely on NPU."
                    ]
                )
            ],
        ],
        [
            HasCPUOnlyOperators(cpu_only_ops=["OP1", "OP2", "OP3"]),
            AdviceCategory.OPERATORS_COMPATIBILITY,
            [
                Advice(
                    [
                        "You have at least 3 operators that is CPU only: "
                        "OP1,OP2,OP3.",
                        "Using operators that are supported by the NPU will "
                        "improve performance.",
                    ]
                )
            ],
        ],
        [
            HasUnsupportedOnNPUOperators(npu_unsupported_ratio=0.4),
            AdviceCategory.OPERATORS_COMPATIBILITY,
            [
                Advice(
                    [
                        "You have 40% of operators that cannot be placed on the NPU.",
                        "For better performance, please review the reasons reported "
                        "in the table, and adjust the model accordingly "
                        "where possible.",
                    ]
                )
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        sram=PerfMetricDiff(100, 150),
                        dram=PerfMetricDiff(100, 50),
                        on_chip_flash=PerfMetricDiff(100, 100),
                        off_chip_flash=PerfMetricDiff(100, 100),
                        npu_total_cycles=PerfMetricDiff(10, 5),
                    ),
                ]
            ),
            AdviceCategory.OPTIMIZATION,
            [
                Advice(
                    [
                        "With the selected optimization (pruning: 0.5)",
                        "- You have achieved 50.00% performance improvement in "
                        "DRAM used (KB)",
                        "- You have achieved 50.00% performance improvement in "
                        "NPU total cycles",
                        "- SRAM used (KB) have degraded by 50.00%",
                        "You can try to push the optimization target higher "
                        "(e.g. pruning: 0.6) "
                        "to check if those results can be further improved.",
                    ]
                )
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[
                            OptimizationSettings("pruning", 0.5, None),
                            OptimizationSettings("clustering", 32, None),
                        ],
                        sram=PerfMetricDiff(100, 150),
                        dram=PerfMetricDiff(100, 50),
                        on_chip_flash=PerfMetricDiff(100, 100),
                        off_chip_flash=PerfMetricDiff(100, 100),
                        npu_total_cycles=PerfMetricDiff(10, 5),
                    ),
                ]
            ),
            AdviceCategory.OPTIMIZATION,
            [
                Advice(
                    [
                        "With the selected optimization (pruning: 0.5, clustering: 32)",
                        "- You have achieved 50.00% performance improvement in "
                        "DRAM used (KB)",
                        "- You have achieved 50.00% performance improvement in "
                        "NPU total cycles",
                        "- SRAM used (KB) have degraded by 50.00%",
                        "You can try to push the optimization target higher "
                        "(e.g. pruning: 0.6 and/or clustering: 16) "
                        "to check if those results can be further improved.",
                    ]
                )
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[
                            OptimizationSettings("clustering", 2, None),
                        ],
                        sram=PerfMetricDiff(100, 150),
                        dram=PerfMetricDiff(100, 50),
                        on_chip_flash=PerfMetricDiff(100, 100),
                        off_chip_flash=PerfMetricDiff(100, 100),
                        npu_total_cycles=PerfMetricDiff(10, 5),
                    ),
                ]
            ),
            AdviceCategory.OPTIMIZATION,
            [
                Advice(
                    [
                        "With the selected optimization (clustering: 2)",
                        "- You have achieved 50.00% performance improvement in "
                        "DRAM used (KB)",
                        "- You have achieved 50.00% performance improvement in "
                        "NPU total cycles",
                        "- SRAM used (KB) have degraded by 50.00%",
                    ]
                )
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        sram=PerfMetricDiff(100, 150),
                        dram=PerfMetricDiff(100, 150),
                        on_chip_flash=PerfMetricDiff(100, 150),
                        off_chip_flash=PerfMetricDiff(100, 150),
                        npu_total_cycles=PerfMetricDiff(10, 100),
                    ),
                ]
            ),
            AdviceCategory.OPTIMIZATION,
            [
                Advice(
                    [
                        "With the selected optimization (pruning: 0.5)",
                        "- DRAM used (KB) have degraded by 50.00%",
                        "- SRAM used (KB) have degraded by 50.00%",
                        "- On chip flash used (KB) have degraded by 50.00%",
                        "- Off chip flash used (KB) have degraded by 50.00%",
                        "- NPU total cycles have degraded by 900.00%",
                        "The performance seems to have degraded after "
                        "applying the selected optimizations, "
                        "try exploring different optimization types/targets.",
                    ]
                )
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        sram=PerfMetricDiff(100, 150),
                        dram=PerfMetricDiff(100, 150),
                        on_chip_flash=PerfMetricDiff(100, 150),
                        off_chip_flash=PerfMetricDiff(100, 150),
                        npu_total_cycles=PerfMetricDiff(10, 100),
                    ),
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.6, None)],
                        sram=PerfMetricDiff(100, 150),
                        dram=PerfMetricDiff(100, 150),
                        on_chip_flash=PerfMetricDiff(100, 150),
                        off_chip_flash=PerfMetricDiff(100, 150),
                        npu_total_cycles=PerfMetricDiff(10, 100),
                    ),
                ]
            ),
            AdviceCategory.OPTIMIZATION,
            [],  # no advice for more than one optimization result
        ],
    ],
)
def test_ethosu_advice_producer(
    dummy_context: Context,
    input_data: DataItem,
    expected_advice: List[Advice],
    advice_category: AdviceCategory,
) -> None:
    """Test Ethos-U Advice producer."""
    producer = EthosUAdviceProducer()

    producer.set_context(dummy_context)
    dummy_context.update(
        advice_category=advice_category, event_handlers=[], config_parameters={}
    )

    producer.produce_advice(input_data)
    advice = producer.get_advice()
    assert advice == expected_advice
