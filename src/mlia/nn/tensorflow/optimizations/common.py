# Copyright (C) 2021-2022, Arm Ltd.
"""Common items for the optimizations module."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class OptimizerConfiguration:
    """Abstract optimizer configuration."""


class Optimizer(ABC):
    """Abstract class for the optimizer."""

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        """Abstract method to return the model instance from the optimizer."""

    @abstractmethod
    def apply_optimization(self) -> None:
        """Abstract method to apply optimization to the model."""

    @abstractmethod
    def optimization_config(self) -> str:
        """Return string representation of the optimization config."""
