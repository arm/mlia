# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collection module for Ethos-U."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import cast

from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import supported_operators
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.performance import P
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.tflite_compat import TFLiteChecker
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.common.optimization import OptimizingPerformaceDataCollector
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import EthosUPerformanceEstimator
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


class EthosUOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path, target_config: EthosUConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.target_config = target_config

    def collect_data(self) -> Operators | TFLiteCompatibilityInfo | None:
        """Collect operator compatibility information."""
        if not is_tflite_model(self.model):
            with log_action("Checking TensorFlow Lite compatibility ..."):
                tflite_checker = TFLiteChecker()
                tflite_compat = tflite_checker.check_compatibility(self.model)

            if not tflite_compat.compatible:
                return tflite_compat

        tflite_model = get_tflite_model(self.model, self.context)

        with log_action("Checking operator compatibility ..."):
            return supported_operators(
                Path(tflite_model.model_path), self.target_config.compiler_options
            )

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_operator_compatibility"


class EthosUPerformance(ContextAwareDataCollector):
    """Collect performance metrics."""

    def __init__(
        self,
        model: Path,
        target_config: EthosUConfiguration,
        backends: list[str] | None = None,
    ) -> None:
        """Init performance data collector."""
        self.model = model
        self.target_config = target_config
        self.backends = backends

    def collect_data(self) -> PerformanceMetrics:
        """Collect model performance metrics."""
        tflite_model = get_tflite_model(self.model, self.context)
        estimator = EthosUPerformanceEstimator(
            self.context,
            self.target_config,
            self.backends,
        )

        return estimator.estimate(tflite_model)

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_performance"


# pylint: disable=too-many-ancestors
class EthosUOptimizationPerformance(OptimizingPerformaceDataCollector):
    """Collect performance metrics for performance optimizations."""

    def create_estimator(self) -> PerformanceEstimator:
        """Create a PerformanceEstimator, to be overridden in subclasses."""
        return EthosUPerformanceEstimator(
            self.context,
            cast(EthosUConfiguration, self.target),
            self.backends,
        )

    def create_optimization_performance_metrics(
        self, original_metrics: P, optimizations_perf_metrics: list[P]
    ) -> Any:
        """Create an optimization metrics object."""
        return OptimizationPerformanceMetrics(
            original_perf_metrics=original_metrics,  # type: ignore
            optimizations_perf_metrics=optimizations_perf_metrics,  # type: ignore
        )

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_model_optimizations"
