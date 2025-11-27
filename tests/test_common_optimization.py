# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the common optimization module."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Callable
from unittest.mock import MagicMock

import pytest

from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.performance import P
from mlia.core.performance import PerformanceEstimator
from mlia.nn.common import Optimizer
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.config import get_keras_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.target.common.optimization import _DEFAULT_OPTIMIZATION_TARGETS
from mlia.target.common.optimization import add_common_optimization_params
from mlia.target.common.optimization import OptimizingDataCollector
from mlia.target.common.optimization import OptimizingPerformaceDataCollector
from mlia.target.common.optimization import parse_augmentations
from mlia.target.config import load_profile
from mlia.target.config import TargetProfile


def _get_mock_optimizer(returned_model: KerasModel | TFLiteModel | Path) -> Optimizer:
    mock_optimizer = MagicMock()
    mock_optimizer.apply_optimization = MagicMock()
    mock_optimizer.get_model = MagicMock(return_value=returned_model)

    return mock_optimizer


def _get_execution_context(
    optimizations_list: list[list[dict]], rewrite_parameters: dict[str, Any]
) -> ExecutionContext:
    return ExecutionContext(
        config_parameters={
            "common_optimizations": {
                "optimizations": optimizations_list,
                "rewrite_parameters": rewrite_parameters,
            }
        }
    )


def _get_opt_settings(
    optimizations_list: list[list[dict]],
) -> list[list[OptimizationSettings]]:
    return [
        [
            OptimizationSettings(
                item.get("optimization_type"),  # type: ignore
                item.get("optimization_target"),  # type: ignore
                item.get("layers_to_optimize"),
                item.get("dataset"),
            )
            for item in opt_configuration
        ]
        for opt_configuration in optimizations_list
    ]


def _get_optimizing_data_collector(
    model: Path,
    optimizations: list[list[dict]],
    rewrite_parameters: dict[str, Any],
) -> OptimizingDataCollector:
    target_profile = MagicMock(spec=TargetProfile)
    collector = OptimizingDataCollector(model, target_profile)
    collector.set_context(_get_execution_context(optimizations, rewrite_parameters))
    return collector


@pytest.mark.parametrize(
    "get_model",
    [  # Given model paths and context, return model to be returned by
        # optimizer.get_model()
        lambda _keras_path, tflite_path, _ctx: tflite_path,
        lambda _keras_path, tflite_path, _ctx: TFLiteModel(tflite_path),
        lambda keras_path, _tflite_path, ctx: get_keras_model(
            keras_path, ctx
        ).get_keras_model(),
    ],
)
def test_optimizing_data_collector_optimize_model(
    get_model: Callable[[Path, Path, Context], Any],
    test_keras_model: Path,
    test_tflite_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OptimizingDataCollector's optimize_model method."""
    rewrite_parameters = {
        "train_params": {"batch_size": 32, "show_progress": False},
        "rewrite_specific_params": None,
    }
    optimizations = [{"optimization_type": "fake", "optimization_target": 42}]
    collector = _get_optimizing_data_collector(
        test_keras_model, [optimizations], rewrite_parameters
    )

    mock_optimizer = _get_mock_optimizer(
        get_model(test_keras_model, test_tflite_model, collector.context)
    )

    monkeypatch.setattr(
        "mlia.target.common.optimization.get_optimizer",
        MagicMock(return_value=mock_optimizer),
    )

    opt_settings = _get_opt_settings([optimizations])[0]
    collector.optimize_model(opt_settings, rewrite_parameters, test_keras_model)

    mock_optimizer.apply_optimization.assert_called_once()  # type: ignore[attr-defined]
    mock_optimizer.get_model.assert_called_once()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "optimizations, expected_err",
    [
        (
            [
                {"optimization_type": "fake", "optimization_target": 42},
            ],
            does_not_raise(),
        ),
        (
            [
                {"optimization_type": "rewrite", "optimization_target": 42},
            ],
            does_not_raise(),
        ),
        (
            [],
            pytest.raises(
                FunctionalityNotSupportedError,
                match="No optimization targets provided",
            ),
        ),
        (
            {"bad": "optimization_params"},
            pytest.raises(
                TypeError,
                match="Optimization parameters expected to be a list.",
            ),
        ),
    ],
)
def test_optimizing_data_collector_collect_data(
    optimizations: list[dict],
    expected_err: Any,
    test_keras_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OptimizingDataCollector's collect_data method."""
    rewrite_parameters = {
        "train_params": {"batch_size": 32, "show_progress": False},
        "rewrite_specific_params": None,
    }
    collector = _get_optimizing_data_collector(
        test_keras_model, [optimizations], rewrite_parameters
    )

    mock_optimize_models = MagicMock()
    monkeypatch.setattr(
        "mlia.target.common.optimization.OptimizingDataCollector.optimize_model",
        mock_optimize_models,
    )

    with expected_err:
        collector.collect_data()
        mock_optimize_models.assert_called_once()


def test_optimizing_data_collector_collect_data_keras_conversion_failed(
    test_keras_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test if FunctionalityNotSupportedError is raised if model
    cannot be converted to keras."""
    rewrite_parameters = {
        "train_params": {"batch_size": 32, "show_progress": False},
        "rewrite_specific_params": None,
    }
    target_profile = MagicMock(spec=TargetProfile)
    optimizations = [{"optimization_type": "fake", "optimization_target": 42}]
    collector = OptimizingDataCollector(test_keras_model, target_profile)
    collector.set_context(_get_execution_context([optimizations], rewrite_parameters))
    monkeypatch.setattr(
        "mlia.target.common.optimization.get_keras_model",
        MagicMock(side_effect=NotImplementedError),
    )
    with pytest.raises(FunctionalityNotSupportedError, match="is not a Keras model"):
        collector.collect_data()


def test_optimizing_performance_data_collector(
    test_keras_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test OptimizingPerformaceDataCollector class."""
    optimized_metrics = {"optimized_metric": 1.0}

    class MyOptimizingPerformaceDataCollector(  # pylint: disable=too-many-ancestors
        OptimizingPerformaceDataCollector
    ):
        """Test child class."""

        def create_estimator(self) -> PerformanceEstimator:
            """Create a PerformanceEstimator."""
            return MagicMock(spec=PerformanceEstimator)

        def create_optimization_performance_metrics(
            self, _original_metrics: P, _optimizations_perf_metrics: list[P]
        ) -> Any:
            """Create an optimization metrics object."""
            return optimized_metrics

    monkeypatch.setattr(
        "mlia.target.common.optimization.estimate_performance",
        MagicMock(return_value=[{"metric": 0.5}, {"metric": 1.0}]),
    )

    collector = MyOptimizingPerformaceDataCollector(
        test_keras_model, MagicMock(spec=TargetProfile)
    )
    perf_metrics = collector.optimize_and_estimate_performance(
        test_keras_model,
        [MagicMock(spec=Optimizer)],
        [[MagicMock(spec=OptimizationSettings)]],
    )

    # Test if optimize_and_estimate_performance returns
    # create_optimization_performance_metrics result
    assert perf_metrics == optimized_metrics


@pytest.mark.parametrize(
    "extra_args, error_to_raise, rewrite_parameter_type",
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
            does_not_raise(),
            type(None),
        ),
        (
            {
                "optimization_targets": [
                    {
                        "optimization_type": "rewrite",
                        "optimization_target": "fully-connected-clustering",
                    }
                ],
                "optimization_profile": load_profile(
                    "src/mlia/resources/optimization_profiles/"
                    "optimization-fully-connected-clustering.toml"
                ),
            },
            does_not_raise(),
            dict,
        ),
        (
            {
                "optimization_targets": [
                    {
                        "optimization_type": "rewrite",
                        "optimization_target": "fully-connected-sparsity",
                    }
                ],
                "optimization_profile": load_profile(
                    "src/mlia/resources/optimization_profiles/"
                    "optimization-fully-connected-pruning.toml"
                ),
            },
            does_not_raise(),
            dict,
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
            type(None),
        ),
        (
            {"optimization_profile": [32, 1e-3, True, 48000, "cosine", 1, 0]},
            pytest.raises(
                TypeError, match="Optimization Parameter values has wrong format."
            ),
            type(None),
        ),
    ],
)
def test_add_common_optimization_params(
    extra_args: dict,
    error_to_raise: Any,
    rewrite_parameter_type: dict | None,
) -> None:
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
            assert advisor_parameters["common_optimizations"]["rewrite_parameters"] == {
                "train_params": None,
                "rewrite_specific_params": None,
            }
        else:
            if not extra_args["optimization_profile"].get("rewrite"):
                assert isinstance(
                    advisor_parameters["common_optimizations"]["rewrite_parameters"][
                        "train_params"
                    ],
                    type(None),
                )
            elif not extra_args["optimization_profile"]["rewrite"].get(
                "training_parameters"
            ):
                assert isinstance(
                    advisor_parameters["common_optimizations"]["rewrite_parameters"][
                        "train_params"
                    ],
                    type(None),
                )
            else:
                assert isinstance(
                    advisor_parameters["common_optimizations"]["rewrite_parameters"][
                        "train_params"
                    ],
                    dict,
                )

            assert isinstance(
                advisor_parameters["common_optimizations"]["rewrite_parameters"][
                    "rewrite_specific_params"
                ],
                rewrite_parameter_type,  # type: ignore
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
