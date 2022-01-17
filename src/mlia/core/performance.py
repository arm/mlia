# Copyright 2021, Arm Ltd.
"""Module for performance estimation."""
from abc import abstractmethod
from typing import Callable
from typing import Generic
from typing import List
from typing import TypeVar


ModelType = TypeVar("ModelType")
PerfMetricsType = TypeVar("PerfMetricsType")


class PerformanceEstimator(Generic[ModelType, PerfMetricsType]):
    """Base class for the performance estimation."""

    @abstractmethod
    def estimate(self, model: ModelType) -> PerfMetricsType:
        """Estimate performance."""


def estimate_performance(
    original_model: ModelType,
    estimator: PerformanceEstimator[ModelType, PerfMetricsType],
    model_transformations: List[Callable[[ModelType], ModelType]],
) -> List[PerfMetricsType]:
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
