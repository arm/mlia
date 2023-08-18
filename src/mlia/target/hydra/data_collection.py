# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Hydra."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.core.performance import P
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.common.optimization import OptimizingPerformaceDataCollector
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.performance import HydraPerformanceEstimator
from mlia.target.hydra.performance import HydraPerformanceMetrics
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


@dataclass
class HydraOptimizationPerformanceMetrics:
    """Optimization performance metrics."""

    original_perf_metrics: HydraPerformanceMetrics
    optimizations_perf_metrics: list[
        tuple[list[OptimizationSettings], HydraPerformanceMetrics]
    ]


class HydraPerformance(ContextAwareDataCollector):
    """Collect performance information."""

    def __init__(self, model: Path, cfg: HydraConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg

    def collect_data(self) -> HydraPerformanceMetrics:
        """Run performance estimator."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        with log_action("Checking performance..."):
            estimator = HydraPerformanceEstimator(self.context, self.cfg)
            metrics = estimator.estimate(self.model)

        return metrics

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "hydra_performance"


# pylint: disable=too-many-ancestors
class HydraOptimizingPerformance(OptimizingPerformaceDataCollector):
    """Optimize and collect performance information."""

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "hydra_optimize_performance"

    def create_estimator(self) -> HydraPerformanceEstimator:
        """Create the estimator object."""
        return HydraPerformanceEstimator(
            self.context, cast(HydraConfiguration, self.target)
        )

    def create_optimization_performance_metrics(
        self, original_metrics: P, optimizations_perf_metrics: list[P]
    ) -> Any:
        """Create an optimization metrics object."""
        return HydraOptimizationPerformanceMetrics(
            original_perf_metrics=original_metrics,  # type: ignore
            optimizations_perf_metrics=optimizations_perf_metrics,  # type: ignore
        )
