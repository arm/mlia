# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Hydra."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

from mlia.backend.argo.performance import ArgoPerformanceEstimator
from mlia.backend.argo.performance import ArgoPerformanceMetrics
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.core.performance import P
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.common.optimization import OptimizingPerformaceDataCollector
from mlia.target.hydra.config import HydraConfiguration
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


@dataclass
class HydraOptimizationPerformanceMetrics:
    """Optimization performance metrics."""

    original_perf_metrics: ArgoPerformanceMetrics
    optimizations_perf_metrics: list[
        tuple[list[OptimizationSettings], ArgoPerformanceMetrics]
    ]


class HydraPerformance(ContextAwareDataCollector):
    """Collect performance information."""

    def __init__(self, model: Path, cfg: HydraConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg

    def collect_data(self) -> ArgoPerformanceMetrics:
        """Run performance estimator."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        with log_action("Checking performance..."):
            estimator = ArgoPerformanceEstimator(
                self.context.output_dir, self.cfg.backend_config
            )
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

    def create_estimator(self) -> ArgoPerformanceEstimator:
        """Create the estimator object."""
        return ArgoPerformanceEstimator(
            self.context.output_dir,
            cast(HydraConfiguration, self.target).backend_config,
        )

    def create_optimization_performance_metrics(
        self, original_metrics: P, optimizations_perf_metrics: list[P]
    ) -> Any:
        """Create an optimization metrics object."""
        return HydraOptimizationPerformanceMetrics(
            original_perf_metrics=original_metrics,  # type: ignore
            optimizations_perf_metrics=optimizations_perf_metrics,  # type: ignore
        )
