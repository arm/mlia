# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collection module for Ethos-U."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from typing import cast

from mlia.backend.vela.compat import supported_operators
from mlia.backend.vela.compat import VelaCompatibilityResult
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
from mlia.target.ethos_u.performance import VelaPerformanceResult
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


class EthosUOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path, target_config: EthosUConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.target_config = target_config

    def collect_data(self) -> VelaCompatibilityResult | TFLiteCompatibilityInfo | None:
        """Collect operator compatibility information."""
        if not is_tflite_model(self.model):
            with log_action("Checking TensorFlow Lite compatibility ..."):
                tflite_checker = TFLiteChecker()
                tflite_compat = tflite_checker.check_compatibility(self.model)

            if not tflite_compat.compatible:
                return tflite_compat

        tflite_model = get_tflite_model(self.model, self.context)

        with log_action("Checking operator compatibility ..."):
            operators = supported_operators(
                Path(tflite_model.model_path), self.target_config.compiler_options
            )

        # Generate standardized output
        target_config = {
            "target": self.target_config.target,
            "mac": self.target_config.mac,
        }
        # Get compiler options for backend configuration
        backend_config = (
            self._get_vela_backend_config(self.target_config.compiler_options)
            if self.target_config.compiler_options
            else {}
        )

        # Clean CLI arguments to use basename for executable
        cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []

        standardized_output = operators.to_standardized_output(
            model_path=Path(tflite_model.model_path),
            target_config=target_config,
            backend_config=backend_config,
            cli_arguments=cli_args,
        )

        return VelaCompatibilityResult(
            legacy_info=operators,
            standardized_output=standardized_output,
        )

    @staticmethod
    def _get_vela_backend_config(compiler_options: Any) -> dict[str, Any]:
        """Extract Vela compiler configuration."""
        return {
            "system_config": compiler_options.system_config,
            "memory_mode": compiler_options.memory_mode,
            "accelerator_config": str(compiler_options.accelerator_config)
            if compiler_options.accelerator_config
            else None,
            "max_block_dependency": compiler_options.max_block_dependency,
            "tensor_allocator": compiler_options.tensor_allocator,
            "optimization_strategy": compiler_options.optimization_strategy,
        }

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

    def collect_data(self) -> PerformanceMetrics | VelaPerformanceResult:
        """Collect model performance metrics."""
        tflite_model = get_tflite_model(self.model, self.context)
        estimator = EthosUPerformanceEstimator(
            self.context,
            self.target_config,
            self.backends,
        )

        perf = estimator.estimate(tflite_model)

        # Wrap with standardized output if Vela backend was used
        if self.backends and "vela" in self.backends:
            try:
                # Get Vela performance metrics from the estimator if available
                if (
                    hasattr(estimator, "vela_perf_metrics")
                    and estimator.vela_perf_metrics
                ):
                    target_config = {
                        "target": self.target_config.target,
                        "mac": self.target_config.mac,
                    }
                    # Get compiler options if available
                    compiler_opts = getattr(estimator, "vela_compiler_options", None)
                    backend_config = (
                        self._get_vela_backend_config(compiler_opts)
                        if compiler_opts
                        else {}
                    )
                    # Clean CLI arguments to use basename for executable
                    cli_args = (
                        [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []
                    )
                    standardized = estimator.vela_perf_metrics.to_standardized_output(
                        model_path=self.model,
                        target_config=target_config,
                        backend_config=backend_config,
                        cli_arguments=cli_args,
                    )
                    return VelaPerformanceResult(
                        legacy_info=perf,
                        standardized_output=standardized,
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to generate standardized output for Vela performance: %s",
                    exc,
                )

        return perf

    @staticmethod
    def _get_vela_backend_config(compiler_options: Any) -> dict[str, Any]:
        """Extract Vela compiler configuration."""
        return {
            "system_config": compiler_options.system_config,
            "memory_mode": compiler_options.memory_mode,
            "accelerator_config": str(compiler_options.accelerator_config)
            if compiler_options.accelerator_config
            else None,
            "max_block_dependency": compiler_options.max_block_dependency,
            "tensor_allocator": compiler_options.tensor_allocator,
            "optimization_strategy": compiler_options.optimization_strategy,
        }

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
