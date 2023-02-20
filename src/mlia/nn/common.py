# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common items for the optimizations module."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from mlia.nn.tensorflow.config import ModelConfiguration


@dataclass
class OptimizerConfiguration:
    """Abstract optimizer configuration."""


class Optimizer(ABC):
    """Abstract class for the optimizer."""

    @abstractmethod
    def get_model(self) -> ModelConfiguration:
        """Abstract method to return the model instance from the optimizer."""

    @abstractmethod
    def apply_optimization(self) -> None:
        """Abstract method to apply optimization to the model."""

    @abstractmethod
    def optimization_config(self) -> str:
        """Return string representation of the optimization config."""
