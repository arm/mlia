# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module select."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import cast

import pytest
import tensorflow as tf

from mlia.core.errors import ConfigurationError
from mlia.nn.rewrite.core.rewrite import RewriteConfiguration
from mlia.nn.rewrite.core.rewrite import RewritingOptimizer
from mlia.nn.rewrite.core.rewrite import TrainingParameters
from mlia.nn.select import get_optimizer
from mlia.nn.select import MultiStageOptimizer
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.optimizations.clustering import Clusterer
from mlia.nn.tensorflow.optimizations.clustering import ClusteringConfiguration
from mlia.nn.tensorflow.optimizations.pruning import Pruner
from mlia.nn.tensorflow.optimizations.pruning import PruningConfiguration


@pytest.mark.parametrize(
    "config, expected_error, expected_type, expected_config",
    [
        (
            OptimizationSettings(
                optimization_type="pruning",
                optimization_target=0.5,
                layers_to_optimize=None,
            ),
            does_not_raise(),
            Pruner,
            "pruning: 0.5",
        ),
        (
            PruningConfiguration(0.5),
            does_not_raise(),
            Pruner,
            "pruning: 0.5",
        ),
        (
            OptimizationSettings(
                optimization_type="clustering",
                optimization_target=32,
                layers_to_optimize=None,
            ),
            does_not_raise(),
            Clusterer,
            "clustering: 32",
        ),
        (
            OptimizationSettings(
                optimization_type="clustering",
                optimization_target=0.5,
                layers_to_optimize=None,
            ),
            pytest.raises(
                ConfigurationError,
                match="Optimization target should be a "
                "positive integer. "
                "Optimization target provided: 0.5",
            ),
            None,
            None,
        ),
        (
            ClusteringConfiguration(32),
            does_not_raise(),
            Clusterer,
            "clustering: 32",
        ),
        (
            OptimizationSettings(
                optimization_type="superoptimization",
                optimization_target="supertarget",  # type: ignore
                layers_to_optimize="all",  # type: ignore
            ),
            pytest.raises(
                ConfigurationError,
                match="Unsupported optimization type: superoptimization",
            ),
            None,
            None,
        ),
        (
            OptimizationSettings(
                optimization_type="",
                optimization_target=0.5,
                layers_to_optimize=None,
            ),
            pytest.raises(
                ConfigurationError,
                match="Optimization type is not provided",
            ),
            None,
            None,
        ),
        (
            "wrong_config",
            pytest.raises(
                Exception,
                match="Unknown optimization configuration wrong_config",
            ),
            None,
            None,
        ),
        (
            OptimizationSettings(
                optimization_type="pruning",
                optimization_target=None,  # type: ignore
                layers_to_optimize=None,
            ),
            pytest.raises(
                Exception,
                match="Optimization target is not provided",
            ),
            None,
            None,
        ),
        (
            [
                OptimizationSettings(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                OptimizationSettings(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                ),
            ],
            does_not_raise(),
            MultiStageOptimizer,
            "pruning: 0.5 - clustering: 32",
        ),
        (
            OptimizationSettings(
                optimization_type="rewrite",
                optimization_target="fully_connected",  # type: ignore
                layers_to_optimize=None,
                dataset=None,
            ),
            does_not_raise(),
            RewritingOptimizer,
            "rewrite: fully_connected",
        ),
        (
            RewriteConfiguration("fully_connected"),
            does_not_raise(),
            RewritingOptimizer,
            "rewrite: fully_connected",
        ),
    ],
)
def test_get_optimizer(
    config: Any,
    expected_error: Any,
    expected_type: type,
    expected_config: str,
    test_keras_model: Path,
    test_tflite_model: Path,
) -> None:
    """Test function get_optimzer."""
    with expected_error:
        if (
            isinstance(config, OptimizationSettings)
            and config.optimization_type == "rewrite"
        ) or isinstance(config, RewriteConfiguration):
            model = test_tflite_model
        else:
            model = tf.keras.models.load_model(str(test_keras_model))
        optimizer = get_optimizer(model, config)
        assert isinstance(optimizer, expected_type)
        assert optimizer.optimization_config() == expected_config


@pytest.mark.parametrize(
    "rewrite_parameters",
    [[None], [{"batch_size": 64, "learning_rate": 0.003}]],
)
@pytest.mark.skip_set_training_steps
def test_get_optimizer_training_parameters(
    rewrite_parameters: list[dict], test_tflite_model: Path
) -> None:
    """Test function get_optimzer with various combinations of parameters."""
    config = OptimizationSettings(
        optimization_type="rewrite",
        optimization_target="fully_connected",  # type: ignore
        layers_to_optimize=None,
        dataset=None,
    )
    optimizer = cast(
        RewritingOptimizer,
        get_optimizer(test_tflite_model, config, list(rewrite_parameters)),
    )

    assert len(rewrite_parameters) == 1

    assert isinstance(
        optimizer.optimizer_configuration.train_params, TrainingParameters
    )
    if not rewrite_parameters[0]:
        assert asdict(TrainingParameters()) == asdict(
            optimizer.optimizer_configuration.train_params
        )
    else:
        assert asdict(TrainingParameters()) | rewrite_parameters[0] == asdict(
            optimizer.optimizer_configuration.train_params
        )


@pytest.mark.parametrize(
    "params, expected_result",
    [
        (
            [],
            [],
        ),
        (
            [("pruning", 0.5)],
            [
                OptimizationSettings(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                )
            ],
        ),
        (
            [("pruning", 0.5), ("clustering", 32)],
            [
                OptimizationSettings(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                OptimizationSettings(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                ),
            ],
        ),
    ],
)
def test_optimization_settings_create_from(
    params: list[tuple[str, float]], expected_result: list[OptimizationSettings]
) -> None:
    """Test creating settings from parsed params."""
    assert OptimizationSettings.create_from(params) == expected_result


@pytest.mark.parametrize(
    "settings, expected_next_target, expected_error",
    [
        [
            OptimizationSettings("clustering", 32, None),
            OptimizationSettings("clustering", 16, None),
            does_not_raise(),
        ],
        [
            OptimizationSettings("clustering", 4, None),
            OptimizationSettings("clustering", 4, None),
            does_not_raise(),
        ],
        [
            OptimizationSettings("clustering", 10, None),
            OptimizationSettings("clustering", 8, None),
            does_not_raise(),
        ],
        [
            OptimizationSettings("pruning", 0.5, None),
            OptimizationSettings("pruning", 0.6, None),
            does_not_raise(),
        ],
        [
            OptimizationSettings("pruning", 0.9, None),
            OptimizationSettings("pruning", 0.9, None),
            does_not_raise(),
        ],
        [
            OptimizationSettings("super_optimization", 42, None),
            None,
            pytest.raises(
                Exception, match="Optimization type super_optimization is unknown."
            ),
        ],
    ],
)
def test_optimization_settings_next_target(
    settings: OptimizationSettings,
    expected_next_target: OptimizationSettings,
    expected_error: Any,
) -> None:
    """Test getting next optimization target."""
    with expected_error:
        assert settings.next_target() == expected_next_target
