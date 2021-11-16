# Copyright 2021, Arm Ltd.
"""Tests for module select."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple

import pytest
import tensorflow as tf
from mlia.optimizations.clustering import Clusterer
from mlia.optimizations.clustering import ClusteringConfiguration
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.pruning import PruningConfiguration
from mlia.optimizations.select import get_optimizer
from mlia.optimizations.select import MultiStageOptimizer
from mlia.optimizations.select import OptimizationSettings


@pytest.mark.parametrize(
    "config, expected_error, expected_type",
    [
        (
            OptimizationSettings(
                optimization_type="pruning",
                optimization_target=0.5,
                layers_to_optimize=None,
            ),
            does_not_raise(),
            Pruner,
        ),
        (
            PruningConfiguration(0.5),
            does_not_raise(),
            Pruner,
        ),
        (
            OptimizationSettings(
                optimization_type="clustering",
                optimization_target=32,
                layers_to_optimize=None,
            ),
            does_not_raise(),
            Clusterer,
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
        ),
        (ClusteringConfiguration(32), does_not_raise(), Clusterer),
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
        ),
        (
            "wrong_config",
            pytest.raises(
                Exception,
                match="Unknown optimization configuration wrong_config",
            ),
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
        ),
    ],
)
def test_get_optimizer(
    config: Any, expected_error: Any, expected_type: type, test_models_path: Path
) -> None:
    """Test function get_optimzer."""
    model_path = str(test_models_path / "simple_model.h5")
    model = tf.keras.models.load_model(model_path)

    with expected_error:
        optimizer = get_optimizer(model, config)
        assert isinstance(optimizer, expected_type)


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
