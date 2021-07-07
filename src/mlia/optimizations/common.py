# Copyright 2021, Arm Ltd.
"""Common items for the optimizations module."""
from abc import ABC
from enum import Enum

import tensorflow as tf


class OptimizerConfiguration(ABC):
    """Abstract optimizer configuration."""


class Optimizer(ABC):
    """Abstract class for the optimizer."""

    def get_model(self) -> tf.keras.Model:
        """Abstract method to return the model instance from the clusterer."""

    def apply_optimization(self) -> None:
        """Abstract method to apply optimization to the model."""


class OptimizationType(Enum):
    """Enumerator for optimization types."""

    Pruning = "pruning"
    Clustering = "clustering"
