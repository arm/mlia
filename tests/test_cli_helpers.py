# Copyright 2022, Arm Ltd.
"""Tests for the helper classes."""
# pylint: disable=no-self-use
from typing import Any
from typing import Dict
from typing import List

import pytest
from mlia.cli.helpers import CLIActionResolver
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings


class TestCliActionResolver:
    """Test cli action resolver."""

    @pytest.mark.parametrize(
        "args, params, expected_result",
        [
            [
                {},
                {"opt_settings": "some_setting"},
                [],
            ],
            [
                {},
                {},
                [
                    "Note: you will need a Keras/TF.saved_model input for that.",
                    "For example: mlia optimization --optimization-type "
                    "pruning,clustering --optimization-target 0.5,32 "
                    "/path/to/keras_model",
                    "For more info: mlia optimization --help",
                ],
            ],
            [
                {"model": "model.h5"},
                {},
                [
                    "For example: mlia optimization --optimization-type "
                    "pruning,clustering --optimization-target 0.5,32 model.h5",
                    "For more info: mlia optimization --help",
                ],
            ],
            [
                {"model": "model.h5"},
                {"opt_settings": [OptimizationSettings("pruning", 0.5, None)]},
                [
                    "For more info: mlia optimization --help",
                    "Optimization command: "
                    "mlia optimization --optimization-type pruning "
                    "--optimization-target 0.5 model.h5",
                ],
            ],
            [
                {"model": "model.h5", "target": "device_target"},
                {"opt_settings": [OptimizationSettings("pruning", 0.5, None)]},
                [
                    "For more info: mlia optimization --help",
                    "Optimization command: "
                    "mlia optimization --optimization-type pruning "
                    "--optimization-target 0.5 --target device_target model.h5",
                ],
            ],
        ],
    )
    def test_apply_optimizations(
        self,
        args: Dict[str, Any],
        params: Dict[str, Any],
        expected_result: List[str],
    ) -> None:
        """Test action resolving for applying optimizations."""
        resolver = CLIActionResolver(args)
        assert resolver.apply_optimizations(**params) == expected_result

    def test_supported_operators_info(self) -> None:
        """Test supported operators info."""
        resolver = CLIActionResolver({})
        assert resolver.supported_operators_info() == [
            "For guidance on supported operators, run: mlia operators "
            "--supported-ops-report",
        ]

    def test_operator_compatibility_details(self) -> None:
        """Test operator compatibility details info."""
        resolver = CLIActionResolver({})
        assert resolver.operator_compatibility_details() == [
            "For more details, run: mlia operators --help"
        ]

    def test_optimization_details(self) -> None:
        """Test optimization details info."""
        resolver = CLIActionResolver({})
        assert resolver.optimization_details() == [
            "For more info, see: mlia optimization --help"
        ]

    @pytest.mark.parametrize(
        "args, expected_result",
        [
            [
                {},
                [],
            ],
            [
                {"model": "model.tflite"},
                [
                    "Check the estimated performance by running the "
                    "following command: ",
                    "mlia performance model.tflite",
                ],
            ],
            [
                {"model": "model.tflite", "target": "device_target"},
                [
                    "Check the estimated performance by running the "
                    "following command: ",
                    "mlia performance --target device_target model.tflite",
                ],
            ],
        ],
    )
    def test_check_performance(
        self, args: Dict[str, Any], expected_result: List[str]
    ) -> None:
        """Test check performance info."""
        resolver = CLIActionResolver(args)
        assert resolver.check_performance() == expected_result

    @pytest.mark.parametrize(
        "args, expected_result",
        [
            [
                {},
                [],
            ],
            [
                {"model": "model.tflite"},
                [
                    "Try running the following command to verify that:",
                    "mlia operators model.tflite",
                ],
            ],
            [
                {"model": "model.tflite", "target": "device_target"},
                [
                    "Try running the following command to verify that:",
                    "mlia operators --target device_target model.tflite",
                ],
            ],
        ],
    )
    def test_check_operator_compatibility(
        self, args: Dict[str, Any], expected_result: List[str]
    ) -> None:
        """Test checking operator compatibility info."""
        resolver = CLIActionResolver(args)
        assert resolver.check_operator_compatibility() == expected_result
