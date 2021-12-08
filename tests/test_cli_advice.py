# Copyright 2021, Arm Ltd.
"""Tests for the advice module."""
from typing import Callable
from typing import List

import pandas as pd
import pytest
from mlia.cli.advice import advice_all_operators_supported
from mlia.cli.advice import advice_all_operators_supported_no_commands
from mlia.cli.advice import advice_increase_operator_compatibility
from mlia.cli.advice import advice_non_npu_operators
from mlia.cli.advice import advice_npu_support
from mlia.cli.advice import advice_optimization
from mlia.cli.advice import advice_optimization_improvement
from mlia.cli.advice import advice_optimization_improvement_extended
from mlia.cli.advice import advice_unsupported_operators
from mlia.cli.advice import AdvisorContext
from mlia.cli.advice import OptimizationResults
from mlia.devices.ethosu.metadata import NpuSupported
from mlia.devices.ethosu.metadata import Operator
from mlia.devices.ethosu.metadata import Operators


@pytest.mark.parametrize(
    "ctx, advice_producer, expected_result",
    [
        (AdvisorContext(), advice_all_operators_supported, []),
        (
            AdvisorContext(
                model="model.tflite",
                operators=Operators(
                    [Operator("operator", "magic operator", NpuSupported(True, []))]
                ),
            ),
            advice_all_operators_supported,
            [
                "You don't have any unsupported operators, "
                "your model will run completely on NPU.",
                "Check the estimated performance by running the following command:",
                "mlia performance model.tflite",
            ],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                device_args={"device": "ethos-u65", "mac": 512},
                operators=Operators(
                    [Operator("operator", "magic operator", NpuSupported(True, []))]
                ),
            ),
            advice_all_operators_supported,
            [
                "You don't have any unsupported operators, "
                "your model will run completely on NPU.",
                "Check the estimated performance by running the following command:",
                "mlia performance --device ethos-u65 --mac 512 model.tflite",
            ],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                device_args={"device": "ethos-u65", "mac": 512},
                operators=Operators(
                    [Operator("operator", "magic operator", NpuSupported(True, []))]
                ),
            ),
            advice_all_operators_supported_no_commands,
            [
                "You don't have any unsupported operators, "
                "your model will run completely on NPU."
            ],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                operators=Operators(
                    [Operator("operator", "magic operator", NpuSupported(False, []))]
                ),
            ),
            advice_all_operators_supported,
            [],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                operators=Operators(
                    [
                        Operator(
                            "operator",
                            "cpu_operator",
                            NpuSupported(False, [("CPU only operator", "")]),
                        )
                    ]
                ),
            ),
            advice_unsupported_operators,
            [
                "You have at least 1 operator that is CPU only: cpu_operator.",
                "Using operators that are supported by the NPU will "
                "improve performance.",
                "For guidance on supported operators, run: mlia operators "
                "--supported-ops-report",
            ],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                operators=Operators(
                    [
                        Operator(
                            "operator",
                            "cpu_operator1",
                            NpuSupported(False, [("CPU only operator", "")]),
                        ),
                        Operator(
                            "operator",
                            "cpu_operator2",
                            NpuSupported(False, [("CPU only operator", "")]),
                        ),
                    ]
                ),
            ),
            advice_unsupported_operators,
            [
                "You have at least 2 operators that is CPU only: "
                "cpu_operator1,cpu_operator2.",
                "Using operators that are supported by the NPU will "
                "improve performance.",
                "For guidance on supported operators, run: mlia operators "
                "--supported-ops-report",
            ],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                operators=Operators(
                    [
                        Operator(
                            "operator",
                            "npu_operator",
                            NpuSupported(True, []),
                        )
                    ]
                ),
            ),
            advice_unsupported_operators,
            [],
        ),
        (AdvisorContext(), advice_unsupported_operators, []),
        (AdvisorContext(), advice_non_npu_operators, []),
        (
            AdvisorContext(
                model="model.tflite",
                operators=Operators(
                    [
                        Operator(
                            "operator",
                            "cpu_operator",
                            NpuSupported(False, [("CPU only operator", "")]),
                        ),
                        Operator(
                            "operator",
                            "npu_operator",
                            NpuSupported(True, []),
                        ),
                    ]
                ),
            ),
            advice_non_npu_operators,
            [
                "You have 50% of operators that cannot be placed on the NPU.",
                "For better performance, please review the reasons reported in the "
                "table, and adjust the model accordingly where possible.",
            ],
        ),
    ],
)
def test_operators_advice(
    ctx: AdvisorContext,
    advice_producer: Callable[[AdvisorContext], List[str]],
    expected_result: List[str],
) -> None:
    """Test operators advice."""
    result = advice_producer(ctx)
    assert result == expected_result


@pytest.mark.parametrize(
    "ctx, advice_producer, expected_result",
    [
        (
            AdvisorContext(model="model.tflite"),
            advice_increase_operator_compatibility,
            [
                "You can improve the inference time by using only operators "
                "that are supported by the NPU.",
                "Try running the following command to verify that:",
                "mlia operators model.tflite",
            ],
        ),
        (
            AdvisorContext(),
            advice_optimization,
            [
                "Check if you can improve the performance by applying "
                "tooling techniques to your model.",
                "Note: you will need a Keras/TF.saved_model input for that.",
                "For example: mlia optimization --optimization-type "
                "pruning,clustering --optimization-target 0.5,32 /path/to/keras_model",
                "For more info: mlia optimization --help",
            ],
        ),
        (
            AdvisorContext(model="model.tflite"),
            advice_optimization,
            [
                "Check if you can improve the performance by applying "
                "tooling techniques to your model.",
                "Note: you will need a Keras/TF.saved_model input for that.",
                "For example: mlia optimization --optimization-type "
                "pruning,clustering --optimization-target 0.5,32 /path/to/keras_model",
                "For more info: mlia optimization --help",
            ],
        ),
        (
            AdvisorContext(model="model.h5"),
            advice_optimization,
            [
                "Check if you can improve the performance by applying "
                "tooling techniques to your model.",
                "For example: mlia optimization --optimization-type "
                "pruning,clustering --optimization-target 0.5,32 model.h5",
                "For more info: mlia optimization --help",
            ],
        ),
        (
            AdvisorContext(
                model="model.tflite",
                device_args={"mac": 32, "optimization_strategy": "Size"},
            ),
            advice_increase_operator_compatibility,
            [
                "You can improve the inference time by using only operators "
                "that are supported by the NPU.",
                "Try running the following command to verify that:",
                "mlia operators --mac 32 --optimization-strategy Size model.tflite",
            ],
        ),
    ],
)
def test_performance_advice(
    ctx: AdvisorContext,
    advice_producer: Callable[[AdvisorContext], List[str]],
    expected_result: List[str],
) -> None:
    """Test performance advice."""
    result = advice_producer(ctx)
    assert result == expected_result


@pytest.mark.parametrize(
    "ctx, advice_producer, expected_result",
    [
        (
            AdvisorContext(),
            advice_npu_support,
            [
                "For better performance, make sure that all the operators of your "
                "final TFLite model are supported by the NPU.",
                "For more details, run: mlia operators --help",
            ],
        ),
        (
            AdvisorContext(
                optimization_results=OptimizationResults(
                    optimizations=[("clustering", 11)],
                    perf_metrics=pd.DataFrame.from_dict(
                        {
                            "Improvement (%)": {
                                "SRAM used (KiB)": 1,
                            }
                        }
                    ),
                )
            ),
            advice_optimization_improvement,
            [
                "With the selected optimization (clustering: 11)",
                "- You have achieved 1.00% performance improvement in SRAM used (KiB)",
                "You can try to push the optimization target higher "
                "(e.g. clustering 8) to check if those results can be further "
                "improved.",
            ],
        ),
        (
            AdvisorContext(
                optimization_results=OptimizationResults(
                    optimizations=[("pruning", 0.5)],
                    perf_metrics=pd.DataFrame.from_dict(
                        {
                            "Improvement (%)": {
                                "SRAM used (KiB)": 1,
                                "DRAM used (KiB)": 2,
                                "On chip flash used (KiB)": 3,
                                "Off chip flash used (KiB)": 4,
                                "NPU total cycles": 5,
                            }
                        }
                    ),
                )
            ),
            advice_optimization_improvement,
            [
                "With the selected optimization (pruning: 0.5)",
                "- You have achieved 1.00% performance improvement in SRAM used (KiB)",
                "- You have achieved 2.00% performance improvement in DRAM used (KiB)",
                "- You have achieved 3.00% performance improvement in On chip flash "
                "used (KiB)",
                "- You have achieved 4.00% performance improvement in Off chip flash "
                "used (KiB)",
                "- You have achieved 5.00% performance improvement in NPU total cycles",
                "You can try to push the optimization target higher "
                "(e.g. pruning 0.6) to check if those results can be further improved.",
            ],
        ),
        (
            AdvisorContext(
                optimization_results=OptimizationResults(
                    optimizations=[("pruning", 0.5)],
                    perf_metrics=pd.DataFrame.from_dict(
                        {
                            "Improvement (%)": {
                                "SRAM used (KiB)": -1,
                                "DRAM used (KiB)": -2,
                                "On chip flash used (KiB)": -3,
                                "Off chip flash used (KiB)": -4,
                                "NPU total cycles": -5,
                            }
                        }
                    ),
                )
            ),
            advice_optimization_improvement,
            [
                "With the selected optimization (pruning: 0.5)",
                "- SRAM used (KiB) have degraded by 1.00%",
                "- DRAM used (KiB) have degraded by 2.00%",
                "- On chip flash used (KiB) have degraded by 3.00%",
                "- Off chip flash used (KiB) have degraded by 4.00%",
                "- NPU total cycles have degraded by 5.00%",
                "The performance seems to have degraded after "
                "applying the selected optimizations, "
                "try exploring different optimization types/targets.",
            ],
        ),
        (
            AdvisorContext(
                model="sample.h5",
                optimization_results=OptimizationResults(
                    optimizations=[("pruning", 0.1), ("clustering", 11)],
                    perf_metrics=pd.DataFrame.from_dict(
                        {
                            "Improvement (%)": {
                                "SRAM used (KiB)": 1,
                            }
                        }
                    ),
                ),
            ),
            advice_optimization_improvement_extended,
            [
                "With the selected optimization (pruning: 0.1 + clustering: 11)",
                "- You have achieved 1.00% performance improvement in SRAM used (KiB)",
                "You can try to push the optimization target higher (e.g. pruning 0.2 "
                "and/or clustering 8) to "
                "check if those results can be further improved.",
                "For more info, see: mlia optimization --help",
                "Optimization command: mlia optimization --optimization-type "
                "pruning,clustering --optimization-target 0.2,8 sample.h5",
            ],
        ),
    ],
)
def test_optmization_advice(
    ctx: AdvisorContext,
    advice_producer: Callable[[AdvisorContext], List[str]],
    expected_result: List[str],
) -> None:
    """Test optimization advice."""
    result = advice_producer(ctx)
    assert result == expected_result
