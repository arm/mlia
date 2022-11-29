# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
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
                    "For example: mlia optimize /path/to/keras_model "
                    "--pruning --clustering "
                    "--pruning-target 0.5 --clustering-target 32",
                    "For more info: mlia optimize --help",
                ],
            ],
            [
                {"model": "model.h5"},
                {},
                [
                    "For example: mlia optimize model.h5 --pruning --clustering "
                    "--pruning-target 0.5 --clustering-target 32",
                    "For more info: mlia optimize --help",
                ],
            ],
            [
                {"model": "model.h5"},
                {"opt_settings": [OptimizationSettings("pruning", 0.5, None)]},
                [
                    "For more info: mlia optimize --help",
                    "Optimization command: "
                    "mlia optimize model.h5 --pruning "
                    "--pruning-target 0.5",
                ],
            ],
            [
                {"model": "model.h5", "target_profile": "target_profile"},
                {"opt_settings": [OptimizationSettings("pruning", 0.5, None)]},
                [
                    "For more info: mlia optimize --help",
                    "Optimization command: "
                    "mlia optimize model.h5 --target-profile target_profile "
                    "--pruning --pruning-target 0.5",
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
    def test_operator_compatibility_details() -> None:
        """Test operator compatibility details info."""
        resolver = CLIActionResolver({})
        assert resolver.operator_compatibility_details() == [
            "For more details, run: mlia check --help"
        ]

    @staticmethod
    def test_optimization_details() -> None:
        """Test optimization details info."""
        resolver = CLIActionResolver({})
        assert resolver.optimization_details() == [
            "For more info, see: mlia optimize --help"
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
                {"model": "model.tflite", "target_profile": "target_profile"},
                [
                    "Check the estimated performance by running the "
                    "following command: ",
                    "mlia check model.tflite "
                    "--target-profile target_profile --performance",
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
                {"model": "model.tflite", "target_profile": "target_profile"},
                [
                    "Try running the following command to verify that:",
                    "mlia check model.tflite --target-profile target_profile",
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
