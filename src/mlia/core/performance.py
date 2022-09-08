# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for performance estimation."""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable
from typing import Generic
from typing import TypeVar


M = TypeVar("M")  # model type
P = TypeVar("P")  # performance metrics


class PerformanceEstimator(Generic[M, P]):
    """Base class for the performance estimation."""

    @abstractmethod
    def estimate(self, model: M) -> P:
        """Estimate performance."""


def estimate_performance(
    original_model: M,
    estimator: PerformanceEstimator[M, P],
    model_transformations: list[Callable[[M], M]],
) -> list[P]:
    """Estimate performance impact.

    This function estimates performance impact on model performance after
    applying provided transformations/optimizations.

    :param original_model: object that represents a model, could be
           instance of the model or path to the model. This depends on
           provided performance estimator.
    :param estimator: performance estimator
    :param model_transformations: list of the callables each of those
           returns object that represents optimized model
    """
    original_metrics = estimator.estimate(original_model)

    optimized_metrics = [
        estimator.estimate(transform(original_model))
        for transform in model_transformations
    ]

    return [original_metrics, *optimized_metrics]
