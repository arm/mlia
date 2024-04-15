# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the common optimization module."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raises
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.core.context import ExecutionContext
from mlia.nn.common import Optimizer
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.target.common.optimization import _DEFAULT_OPTIMIZATION_TARGETS
from mlia.target.common.optimization import add_common_optimization_params
from mlia.target.common.optimization import OptimizingDataCollector
from mlia.target.common.optimization import parse_augmentations
from mlia.target.config import load_profile
from mlia.target.config import TargetProfile


class FakeOptimizer(Optimizer):
    """Optimizer for testing purposes."""

    def __init__(self, optimized_model_path: Path) -> None:
        """Initialize."""
        super().__init__()
        self.optimized_model_path = optimized_model_path
        self.invocation_count = 0

    def apply_optimization(self) -> None:
        """Count the invocations."""
        self.invocation_count += 1

    def get_model(self) -> TFLiteModel:
        """Return optimized model."""
        return TFLiteModel(self.optimized_model_path)

    def optimization_config(self) -> str:
        """Return something: doesn't matter, not used."""
        return ""


def test_optimizing_data_collector(
    test_keras_model: Path,
    test_tflite_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OptimizingDataCollector, base support for various targets."""
    optimizations = [
        [
            {"optimization_type": "fake", "optimization_target": 42},
        ]
    ]
    training_parameters = {"batch_size": 32, "show_progress": False}
    context = ExecutionContext(
        config_parameters={
            "common_optimizations": {
                "optimizations": optimizations,
                "training_parameters": training_parameters,
            }
        }
    )

    target_profile = MagicMock(spec=TargetProfile)

    fake_optimizer = FakeOptimizer(test_tflite_model)

    monkeypatch.setattr(
        "mlia.target.common.optimization.get_optimizer",
        MagicMock(return_value=fake_optimizer),
    )

    collector = OptimizingDataCollector(test_keras_model, target_profile)

    optimize_model_mock = MagicMock(side_effect=collector.optimize_model)
    monkeypatch.setattr(
        "mlia.target.common.optimization.OptimizingDataCollector.optimize_model",
        optimize_model_mock,
    )
    opt_settings = [
        [
            OptimizationSettings(
                item.get("optimization_type"),  # type: ignore
                item.get("optimization_target"),  # type: ignore
                item.get("layers_to_optimize"),  # type: ignore
                item.get("dataset"),  # type: ignore
            )
            for item in opt_configuration
        ]
        for opt_configuration in optimizations
    ]

    collector.set_context(context)
    collector.collect_data()
    assert optimize_model_mock.call_args.args[0] == opt_settings[0]
    assert optimize_model_mock.call_args.args[1] == training_parameters
    assert fake_optimizer.invocation_count == 1


@pytest.mark.parametrize(
    "extra_args, error_to_raise",
    [
        (
            {
                "optimization_targets": [
                    {
                        "optimization_type": "pruning",
                        "optimization_target": 0.5,
                        "layers_to_optimize": None,
                    }
                ],
            },
            does_not_raises(),
        ),
        (
            {
                "optimization_profile": load_profile(
                    "src/mlia/resources/optimization_profiles/optimization.toml"
                )
            },
            does_not_raises(),
        ),
        (
            {
                "optimization_targets": {
                    "optimization_type": "pruning",
                    "optimization_target": 0.5,
                    "layers_to_optimize": None,
                },
            },
            pytest.raises(
                TypeError, match="Optimization targets value has wrong format."
            ),
        ),
        (
            {"optimization_profile": [32, 1e-3, True, 48000, "cosine", 1, 0]},
            pytest.raises(
                TypeError, match="Training Parameter values has wrong format."
            ),
        ),
    ],
)
def test_add_common_optimization_params(extra_args: dict, error_to_raise: Any) -> None:
    """Test to check that optimization_targets and optimization_profiles are
    correctly parsed."""
    advisor_parameters: dict = {}

    with error_to_raise:
        add_common_optimization_params(advisor_parameters, extra_args)
        if not extra_args.get("optimization_targets"):
            assert advisor_parameters["common_optimizations"]["optimizations"] == [
                _DEFAULT_OPTIMIZATION_TARGETS
            ]
        else:
            assert advisor_parameters["common_optimizations"]["optimizations"] == [
                extra_args["optimization_targets"]
            ]

        if not extra_args.get("optimization_profile"):
            assert (
                advisor_parameters["common_optimizations"]["training_parameters"]
                is None
            )
        else:
            assert (
                advisor_parameters["common_optimizations"]["training_parameters"]
                == extra_args["optimization_profile"]["training"]
            )


@pytest.mark.parametrize(
    "augmentations, expected_output",
    [
        (
            {"gaussian_strength": 1.0, "mixup_strength": 1.0},
            (1.0, 1.0),
        ),
        (
            {"gaussian_strength": 1.0},
            (None, 1.0),
        ),
        (
            {"Wrong param": 1.0, "mixup_strength": 1.0},
            (1.0, None),
        ),
        (
            {"Wrong param1": 1.0, "Wrong param2": 1.0},
            (None, None),
        ),
        (
            "gaussian",
            (None, 1.0),
        ),
        (
            "mix_gaussian_large",
            (2.0, 1.0),
        ),
        (
            "not in presets",
            (None, None),
        ),
        (
            {"gaussian_strength": 1.0, "mixup_strength": 1.0, "mix2": 1.0},
            (1.0, 1.0),
        ),
        (
            {"gaussian_strength": "not a float", "mixup_strength": 1.0},
            (1.0, None),
        ),
        (
            None,
            (None, None),
        ),
    ],
)
def test_parse_augmentations(
    augmentations: dict | str | None, expected_output: tuple
) -> None:
    """Check that augmentation parameters in optimization_profiles are
    correctly parsed."""

    augmentation_output = parse_augmentations(augmentations)
    assert augmentation_output == expected_output
