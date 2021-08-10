# Copyright 2021, Arm Ltd.
"""Tests for the advice module."""
from typing import Callable
from typing import List

import pytest
from mlia.cli.advice import advice_all_operators_supported
from mlia.cli.advice import advice_non_npu_operators
from mlia.cli.advice import advice_unsupported_operators
from mlia.cli.advice import AdvisorContext
from mlia.metadata import NpuSupported
from mlia.metadata import Operator
from mlia.metadata import Operators


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
