# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the helper classes."""
from __future__ import annotations

from typing import Any

import pytest

from mlia.cli.helpers import CLIActionResolver
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings


class TestCliActionResolver:
    """Test cli action resolver."""

    @staticmethod
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
                    "Note: you will need a Keras model for that.",
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
                {"model": "model.h5", "target_profile": "target_profile"},
                {"opt_settings": [OptimizationSettings("pruning", 0.5, None)]},
                [
                    "For more info: mlia optimization --help",
                    "Optimization command: "
                    "mlia optimization --optimization-type pruning "
                    "--optimization-target 0.5 "
                    "--target-profile target_profile model.h5",
                ],
            ],
        ],
    )
    def test_apply_optimizations(
        args: dict[str, Any],
        params: dict[str, Any],
        expected_result: list[str],
    ) -> None:
        """Test action resolving for applying optimizations."""
        resolver = CLIActionResolver(args)
        assert resolver.apply_optimizations(**params) == expected_result

    @staticmethod
    def test_supported_operators_info() -> None:
        """Test supported operators info."""
        resolver = CLIActionResolver({})
        assert resolver.supported_operators_info() == [
            "For guidance on supported operators, run: mlia operators "
            "--supported-ops-report",
        ]

    @staticmethod
    def test_operator_compatibility_details() -> None:
        """Test operator compatibility details info."""
        resolver = CLIActionResolver({})
        assert resolver.operator_compatibility_details() == [
            "For more details, run: mlia operators --help"
        ]

    @staticmethod
    def test_optimization_details() -> None:
        """Test optimization details info."""
        resolver = CLIActionResolver({})
        assert resolver.optimization_details() == [
            "For more info, see: mlia optimization --help"
        ]

    @staticmethod
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
                {"model": "model.tflite", "target_profile": "target_profile"},
                [
                    "Check the estimated performance by running the "
                    "following command: ",
                    "mlia performance --target-profile target_profile model.tflite",
                ],
            ],
        ],
    )
    def test_check_performance(
        args: dict[str, Any], expected_result: list[str]
    ) -> None:
        """Test check performance info."""
        resolver = CLIActionResolver(args)
        assert resolver.check_performance() == expected_result

    @staticmethod
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
                {"model": "model.tflite", "target_profile": "target_profile"},
                [
                    "Try running the following command to verify that:",
                    "mlia operators --target-profile target_profile model.tflite",
                ],
            ],
        ],
    )
    def test_check_operator_compatibility(
        args: dict[str, Any], expected_result: list[str]
    ) -> None:
        """Test checking operator compatibility info."""
        resolver = CLIActionResolver(args)
        assert resolver.check_operator_compatibility() == expected_result
