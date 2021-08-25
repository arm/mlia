# Copyright 2021, Arm Ltd.
"""Multi stage optimizations."""
from typing import List

import tensorflow as tf
from mlia.optimizations.clustering import Clusterer
from mlia.optimizations.clustering import ClusteringConfiguration
from mlia.optimizations.common import Optimizer
from mlia.optimizations.common import OptimizerConfiguration
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.pruning import PruningConfiguration


class MultiStageOptimizer(Optimizer):
    """Optimizer with multiply stages."""

    def __init__(
        self,
        model: tf.keras.Model,
        optimizations: List[OptimizerConfiguration],
    ) -> None:
        """Init MultiStageOptimizer instance."""
        self.model = model
        self.optimizations = optimizations

    def get_model(self) -> tf.keras.Model:
        """Return optimized model."""
        return self.model

    def apply_optimization(self) -> None:
        """Apply optimization to the model."""
        for config in self.optimizations:
            optimizer = get_optimizer(self.model, config)
            optimizer.apply_optimization()
            self.model = optimizer.get_model()


def get_optimizer(model: tf.keras.Model, config: OptimizerConfiguration) -> Optimizer:
    """Get optimizer for provided configuration."""
    if isinstance(config, PruningConfiguration):
        return Pruner(model, config)

    if isinstance(config, ClusteringConfiguration):
        return Clusterer(model, config)

    raise Exception(f"Unknown optimization configuration {config}")
