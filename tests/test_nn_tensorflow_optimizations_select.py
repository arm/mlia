# Copyright 2021, Arm Ltd.
"""Tests for module select."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple

import pytest
import tensorflow as tf
from mlia.nn.tensorflow.optimizations.clustering import Clusterer
from mlia.nn.tensorflow.optimizations.clustering import ClusteringConfiguration
from mlia.nn.tensorflow.optimizations.pruning import Pruner
from mlia.nn.tensorflow.optimizations.pruning import PruningConfiguration
from mlia.nn.tensorflow.optimizations.select import get_optimizer
from mlia.nn.tensorflow.optimizations.select import MultiStageOptimizer
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings


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
                Exception,
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
                Exception,
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
                Exception,
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
    ],
)
def test_get_optimizer(
    config: Any,
    expected_error: Any,
    expected_type: type,
    expected_config: str,
    test_models_path: Path,
) -> None:
    """Test function get_optimzer."""
    model_path = str(test_models_path / "simple_model.h5")
    model = tf.keras.models.load_model(model_path)

    with expected_error:
        optimizer = get_optimizer(model, config)
        assert isinstance(optimizer, expected_type)
        assert optimizer.optimization_config() == expected_config


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
    params: List[Tuple[str, float]], expected_result: List[OptimizationSettings]
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
                Exception, match="Unknown optimization type super_optimization"
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
