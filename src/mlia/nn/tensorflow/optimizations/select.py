# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for optimization selection."""
from __future__ import annotations

import math
from typing import NamedTuple

import tensorflow as tf

from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.optimizations.clustering import Clusterer
from mlia.nn.tensorflow.optimizations.clustering import ClusteringConfiguration
from mlia.nn.tensorflow.optimizations.common import Optimizer
from mlia.nn.tensorflow.optimizations.common import OptimizerConfiguration
from mlia.nn.tensorflow.optimizations.pruning import Pruner
from mlia.nn.tensorflow.optimizations.pruning import PruningConfiguration
from mlia.utils.types import is_list_of


class OptimizationSettings(NamedTuple):
    """Optimization settings."""

    optimization_type: str
    optimization_target: int | float
    layers_to_optimize: list[str] | None

    @staticmethod
    def create_from(
        optimizer_params: list[tuple[str, float]],
        layers_to_optimize: list[str] | None = None,
    ) -> list[OptimizationSettings]:
        """Create optimization settings from the provided parameters."""
        return [
            OptimizationSettings(
                optimization_type=opt_type,
                optimization_target=opt_target,
                layers_to_optimize=layers_to_optimize,
            )
            for opt_type, opt_target in optimizer_params
        ]

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.optimization_type}: {self.optimization_target}"

    def next_target(self) -> OptimizationSettings:
        """Return next optimization target."""
        if self.optimization_type == "pruning":
            next_target = round(min(self.optimization_target + 0.1, 0.9), 2)
            return OptimizationSettings(
                self.optimization_type, next_target, self.layers_to_optimize
            )

        if self.optimization_type == "clustering":
            # return next lowest power of two for clustering
            next_target = math.log(self.optimization_target, 2)
            if next_target.is_integer():
                next_target -= 1

            next_target = max(int(2 ** int(next_target)), 4)
            return OptimizationSettings(
                self.optimization_type, next_target, self.layers_to_optimize
            )

        raise ValueError(f"Optimization type {self.optimization_type} is unknown.")


class MultiStageOptimizer(Optimizer):
    """Optimizer with multiply stages."""

    def __init__(
        self,
        model: tf.keras.Model,
        optimizations: list[OptimizerConfiguration],
    ) -> None:
        """Init MultiStageOptimizer instance."""
        self.model = model
        self.optimizations = optimizations

    def optimization_config(self) -> str:
        """Return string representation of the optimization config."""
        return " - ".join(str(opt) for opt in self.optimizations)

    def get_model(self) -> tf.keras.Model:
        """Return optimized model."""
        return self.model

    def apply_optimization(self) -> None:
        """Apply optimization to the model."""
        for config in self.optimizations:
            optimizer = get_optimizer(self.model, config)
            optimizer.apply_optimization()
            self.model = optimizer.get_model()


def get_optimizer(
    model: tf.keras.Model | KerasModel,
    config: OptimizerConfiguration | OptimizationSettings | list[OptimizationSettings],
) -> Optimizer:
    """Get optimizer for provided configuration."""
    if isinstance(model, KerasModel):
        model = model.get_keras_model()

    if isinstance(config, PruningConfiguration):
        return Pruner(model, config)

    if isinstance(config, ClusteringConfiguration):
        return Clusterer(model, config)

    if isinstance(config, OptimizationSettings) or is_list_of(
        config, OptimizationSettings
    ):
        return _get_optimizer(model, config)  # type: ignore

    raise ConfigurationError(f"Unknown optimization configuration {config}")


def _get_optimizer(
    model: tf.keras.Model,
    optimization_settings: OptimizationSettings | list[OptimizationSettings],
) -> Optimizer:
    if isinstance(optimization_settings, OptimizationSettings):
        optimization_settings = [optimization_settings]

    optimizer_configs = []
    for opt_type, opt_target, layers_to_optimize in optimization_settings:
        _check_optimizer_params(opt_type, opt_target)

        opt_config = _get_optimizer_configuration(
            opt_type, opt_target, layers_to_optimize
        )
        optimizer_configs.append(opt_config)

    if len(optimizer_configs) == 1:
        return get_optimizer(model, optimizer_configs[0])

    return MultiStageOptimizer(model, optimizer_configs)


def _get_optimizer_configuration(
    optimization_type: str,
    optimization_target: int | float,
    layers_to_optimize: list[str] | None = None,
) -> OptimizerConfiguration:
    """Get optimizer configuration for provided parameters."""
    _check_optimizer_params(optimization_type, optimization_target)

    opt_type = optimization_type.lower()
    if opt_type == "pruning":
        return PruningConfiguration(optimization_target, layers_to_optimize)

    if opt_type == "clustering":
        # make sure an integer is given as clustering target
        if optimization_target == int(optimization_target):
            return ClusteringConfiguration(int(optimization_target), layers_to_optimize)

        raise ConfigurationError(
            "Optimization target should be a positive integer. "
            f"Optimization target provided: {optimization_target}"
        )

    raise ConfigurationError(f"Unsupported optimization type: {optimization_type}")


def _check_optimizer_params(
    optimization_type: str, optimization_target: int | float
) -> None:
    """Check optimizer params."""
    if not optimization_target:
        raise ConfigurationError("Optimization target is not provided")

    if not optimization_type:
        raise ConfigurationError("Optimization type is not provided")
