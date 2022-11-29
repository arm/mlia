# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U advice generation."""
from __future__ import annotations

import pytest

from mlia.cli.helpers import CLIActionResolver
from mlia.core.advice_generation import Advice
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import ExecutionContext
from mlia.core.helpers import ActionResolver
from mlia.core.helpers import APIActionResolver
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.target.ethos_u.advice_generation import EthosUAdviceProducer
from mlia.target.ethos_u.advice_generation import EthosUStaticAdviceProducer
from mlia.target.ethos_u.data_analysis import AllOperatorsSupportedOnNPU
from mlia.target.ethos_u.data_analysis import HasCPUOnlyOperators
from mlia.target.ethos_u.data_analysis import HasUnsupportedOnNPUOperators
from mlia.target.ethos_u.data_analysis import OptimizationDiff
from mlia.target.ethos_u.data_analysis import OptimizationResults
from mlia.target.ethos_u.data_analysis import PerfMetricDiff


@pytest.mark.parametrize(
    "input_data, advice_category, action_resolver, expected_advice",
    [
        [
            AllOperatorsSupportedOnNPU(),
            {AdviceCategory.COMPATIBILITY},
            APIActionResolver(),
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
            AllOperatorsSupportedOnNPU(),
            {AdviceCategory.COMPATIBILITY},
            CLIActionResolver(
                {
                    "target_profile": "sample_target",
                    "model": "sample_model.tflite",
                }
            ),
            [
                Advice(
                    [
                        "You don't have any unsupported operators, your model will "
                        "run completely on NPU.",
                        "Check the estimated performance by running the "
                        "following command: ",
                        "mlia check sample_model.tflite --target-profile sample_target "
                        "--performance",
                    ]
                )
            ],
        ],
        [
            HasCPUOnlyOperators(cpu_only_ops=["OP1", "OP2", "OP3"]),
            {AdviceCategory.COMPATIBILITY},
            APIActionResolver(),
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
            HasCPUOnlyOperators(cpu_only_ops=["OP1", "OP2", "OP3"]),
            {AdviceCategory.COMPATIBILITY},
            CLIActionResolver({}),
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
            {AdviceCategory.COMPATIBILITY},
            APIActionResolver(),
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
            HasUnsupportedOnNPUOperators(npu_unsupported_ratio=0.4),
            {AdviceCategory.COMPATIBILITY},
            CLIActionResolver({}),
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
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
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
                ),
                Advice(
                    [
                        "The applied tooling techniques have an impact "
                        "on accuracy. Additional hyperparameter tuning may be required "
                        "after any optimization."
                    ]
                ),
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            CLIActionResolver({"model": "sample_model.h5"}),
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
                        "For more info: mlia optimize --help",
                        "Optimization command: "
                        "mlia optimize sample_model.h5 --pruning "
                        "--pruning-target 0.6",
                    ]
                ),
                Advice(
                    [
                        "The applied tooling techniques have an impact "
                        "on accuracy. Additional hyperparameter tuning may be required "
                        "after any optimization."
                    ]
                ),
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
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
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
                ),
                Advice(
                    [
                        "The applied tooling techniques have an impact "
                        "on accuracy. Additional hyperparameter tuning may be required "
                        "after any optimization."
                    ]
                ),
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[
                            OptimizationSettings("clustering", 2, None),
                        ],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
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
                ),
                Advice(
                    [
                        "The applied tooling techniques have an impact "
                        "on accuracy. Additional hyperparameter tuning may be required "
                        "after any optimization."
                    ]
                ),
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 150),
                            "on_chip_flash": PerfMetricDiff(100, 150),
                            "off_chip_flash": PerfMetricDiff(100, 150),
                            "npu_total_cycles": PerfMetricDiff(10, 100),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
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
                ),
                Advice(
                    [
                        "The applied tooling techniques have an impact "
                        "on accuracy. Additional hyperparameter tuning may be required "
                        "after any optimization."
                    ]
                ),
            ],
        ],
        [
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 150),
                            "on_chip_flash": PerfMetricDiff(100, 150),
                            "off_chip_flash": PerfMetricDiff(100, 150),
                            "npu_total_cycles": PerfMetricDiff(10, 100),
                        },
                    ),
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.6, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 150),
                            "on_chip_flash": PerfMetricDiff(100, 150),
                            "off_chip_flash": PerfMetricDiff(100, 150),
                            "npu_total_cycles": PerfMetricDiff(10, 100),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [],  # no advice for more than one optimization result
        ],
    ],
)
def test_ethosu_advice_producer(
    tmpdir: str,
    input_data: DataItem,
    expected_advice: list[Advice],
    advice_category: set[AdviceCategory] | None,
    action_resolver: ActionResolver,
) -> None:
    """Test Ethos-U Advice producer."""
    producer = EthosUAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        working_dir=tmpdir,
        action_resolver=action_resolver,
    )

    producer.set_context(context)
    producer.produce_advice(input_data)

    assert producer.get_advice() == expected_advice


@pytest.mark.parametrize(
    "advice_category, action_resolver, expected_advice",
    [
        [
            {AdviceCategory.COMPATIBILITY, AdviceCategory.PERFORMANCE},
            None,
            [],
        ],
        [
            {AdviceCategory.COMPATIBILITY},
            None,
            [],
        ],
        [
            {AdviceCategory.PERFORMANCE},
            APIActionResolver(),
            [
                Advice(
                    [
                        "You can improve the inference time by using only operators "
                        "that are supported by the NPU.",
                    ]
                ),
                Advice(
                    [
                        "Check if you can improve the performance by applying "
                        "tooling techniques to your model."
                    ]
                ),
            ],
        ],
        [
            {AdviceCategory.PERFORMANCE},
            CLIActionResolver(
                {"model": "test_model.h5", "target_profile": "sample_target"}
            ),
            [
                Advice(
                    [
                        "You can improve the inference time by using only operators "
                        "that are supported by the NPU.",
                        "Try running the following command to verify that:",
                        "mlia check test_model.h5 --target-profile sample_target",
                    ]
                ),
                Advice(
                    [
                        "Check if you can improve the performance by applying "
                        "tooling techniques to your model.",
                        "For example: mlia optimize test_model.h5 "
                        "--pruning --clustering "
                        "--pruning-target 0.5 --clustering-target 32",
                        "For more info: mlia optimize --help",
                    ]
                ),
            ],
        ],
        [
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [
                Advice(
                    [
                        "For better performance, make sure that all the operators "
                        "of your final TensorFlow Lite model are supported by the NPU.",
                    ]
                )
            ],
        ],
        [
            {AdviceCategory.OPTIMIZATION},
            CLIActionResolver({"model": "test_model.h5"}),
            [
                Advice(
                    [
                        "For better performance, make sure that all the operators "
                        "of your final TensorFlow Lite model are supported by the NPU.",
                        "For more details, run: mlia check --help",
                    ]
                )
            ],
        ],
    ],
)
def test_ethosu_static_advice_producer(
    tmpdir: str,
    advice_category: set[AdviceCategory] | None,
    action_resolver: ActionResolver,
    expected_advice: list[Advice],
) -> None:
    """Test static advice generation."""
    producer = EthosUStaticAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        working_dir=tmpdir,
        action_resolver=action_resolver,
    )
    producer.set_context(context)
    assert producer.get_advice() == expected_advice
