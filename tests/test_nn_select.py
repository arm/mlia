# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module select."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import cast

import pytest
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from mlia.core.errors import ConfigurationError
from mlia.nn.rewrite.core.rewrite import RewriteConfiguration
from mlia.nn.rewrite.core.rewrite import RewritingOptimizer
from mlia.nn.rewrite.core.rewrite import TrainingParameters
from mlia.nn.select import get_optimizer
from mlia.nn.select import MultiStageOptimizer
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.config import TFLiteModel
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
            [
                OptimizationSettings(
                    optimization_type="rewrite",
                    optimization_target={"bad_type": 12},  # type: ignore
                    layers_to_optimize=None,
                ),
                OptimizationSettings(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                ),
            ],
            pytest.raises(
                ConfigurationError,
                match="Optimization target should be a string indicating a"
                "choice from rewrite library. ",
            ),
            MultiStageOptimizer,
            "pruning: 0.5 - clustering: 32",
        ),
        (
            OptimizationSettings(
                optimization_type="rewrite",
                optimization_target="fully-connected",  # type: ignore
                layers_to_optimize=None,
                dataset=None,
            ),
            does_not_raise(),
            RewritingOptimizer,
            "rewrite: fully-connected",
        ),
        (
            RewriteConfiguration("fully-connected"),
            does_not_raise(),
            RewritingOptimizer,
            "rewrite: fully-connected",
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
            model = keras.models.load_model(str(test_keras_model))
        optimizer = get_optimizer(
            model, config, {"train_params": None, "rewrite_specific_params": None}
        )
        assert isinstance(optimizer, expected_type)
        assert optimizer.optimization_config() == expected_config
        assert optimizer.get_model() is not None


def test_get_optimizer_tflite(
    test_tflite_model: Path,
) -> None:
    """Test get_optimizer for a TFLiteModel input model"""
    config = OptimizationSettings(
        optimization_type="clustering",
        optimization_target=32,
        layers_to_optimize=None,
    )
    tflite_model = TFLiteModel(test_tflite_model)
    optimizer = get_optimizer(
        tflite_model, config, {"train_params": None, "rewrite_specific_params": None}
    )
    assert isinstance(optimizer, Clusterer)
    assert optimizer.optimization_config() == "clustering: 32"
    assert optimizer.get_model() is not None


# pylint: disable=line-too-long
@pytest.mark.parametrize(
    "rewrite_parameters, optimization_target",
    [
        [
            {"train_params": None, "rewrite_specific_params": None},
            "fully-connected-clustering",
        ],
        [
            {
                "train_params": None,
                "rewrite_specific_params": {
                    "num_clusters": 5,
                    "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization(
                        "CentroidInitialization.LINEAR"
                    ),
                },
            },
            "fully-connected-clustering",
        ],
        [
            {"train_params": None, "rewrite_specific_params": None},
            "fully-connected",
        ],
        [
            {"train_params": {"batch_size": 16}, "rewrite_specific_params": None},
            "fully-connected",
        ],
    ],
)
# pylint: enable=line-too-long
@pytest.mark.skip_set_training_steps
def test_get_optimizer_training_parameters(
    rewrite_parameters: dict,
    optimization_target: str,
    test_tflite_model: Path,
) -> None:
    """Test function get_optimzer with various combinations of parameters."""
    config = OptimizationSettings(
        optimization_type="rewrite",
        optimization_target=optimization_target,  # type: ignore
        layers_to_optimize=None,
        dataset=None,
    )
    optimizer = cast(
        RewritingOptimizer,
        get_optimizer(test_tflite_model, config, rewrite_parameters),
    )
    assert len(list(rewrite_parameters.items())) == 2
    if rewrite_parameters.get("rewrite_specific_params"):
        assert isinstance(
            rewrite_parameters["rewrite_specific_params"],
            type(optimizer.optimizer_configuration.rewrite_specific_params),
        )
        assert (
            optimizer.optimizer_configuration.rewrite_specific_params
            == rewrite_parameters["rewrite_specific_params"]
        )

    assert isinstance(
        optimizer.optimizer_configuration.train_params, TrainingParameters
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
        [
            OptimizationSettings("rewrite", 32, None),
            OptimizationSettings("rewrite", 32, None),
            does_not_raise(),
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


def test_apply_optimization(
    test_keras_model: Path,
) -> None:
    """Test apply_optimization in MultiStageOptimizer"""
    config = [
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
    ]
    model = keras.models.load_model(str(test_keras_model))
    optimizer = get_optimizer(
        model, config, {"train_params": None, "rewrite_specific_params": None}
    )
    assert isinstance(optimizer, MultiStageOptimizer)
    optimizer.apply_optimization()
