# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collection module for Ethos-U."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import supported_operators
from mlia.core.context import Context
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.performance import estimate_performance
from mlia.nn.select import get_optimizer
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.config import get_keras_model
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.tflite_compat import TFLiteChecker
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.nn.tensorflow.utils import save_keras_model
from mlia.nn.tensorflow.utils import save_tflite_model
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import EthosUPerformanceEstimator
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.utils.logging import log_action
from mlia.utils.types import is_list_of

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


class OptimizeModel:
    """Helper class for model optimization."""

    def __init__(
        self, context: Context, opt_settings: list[OptimizationSettings]
    ) -> None:
        """Init helper."""
        self.context = context
        self.opt_settings = opt_settings

    def __call__(self, keras_model: KerasModel) -> Any:
        """Run optimization."""
        optimizer = get_optimizer(keras_model, self.opt_settings)

        opts_as_str = ", ".join(str(opt) for opt in self.opt_settings)
        logger.info("Applying model optimizations - [%s]", opts_as_str)
        optimizer.apply_optimization()

        model = optimizer.get_model()

        if isinstance(model, Path):
            return model

        if isinstance(model, TFLiteModel):
            model_path = self.context.get_model_path("optimized_model.tflite")
            with open(model.model_path, "rb") as file_handle:
                model_data = bytearray(file_handle.read())
            save_tflite_model(model_data, model_path)
            return TFLiteModel(model_path)

        model_path = self.context.get_model_path("optimized_model.h5")
        save_keras_model(model, model_path)
        return KerasModel(model_path)


class EthosUOptimizationPerformance(ContextAwareDataCollector):
    """Collect performance metrics for the optimizations."""

    def __init__(
        self,
        model: Path,
        target_config: EthosUConfiguration,
        optimizations: list[list[dict]],
        backends: list[str] | None = None,
    ) -> None:
        """Init performance optimizations data collector."""
        self.model = model
        self.target = target_config
        self.optimizations = optimizations
        self.backends = backends

    def collect_data(self) -> OptimizationPerformanceMetrics | None:
        """Collect performance metrics for the optimizations."""
        logger.info("Estimate performance ...")

        if not self.optimizations:
            raise FunctionalityNotSupportedError(
                reason="Unable to estimate model optimizations impact",
                description="No optimization targets provided",
            )

        opt_settings = self._parse_optimization_params(self.optimizations)

        if opt_settings[0][0].optimization_type != "rewrite":
            try:
                model = get_keras_model(self.model, self.context)
            except NotImplementedError as err:
                raise FunctionalityNotSupportedError(
                    reason="Unable to run model optimizations",
                    description=f"{self.model} is not a Keras model and "
                    "could not be converted to a Keras model",
                ) from err
        else:
            model = self.model  # type: ignore

        optimizers = [OptimizeModel(self.context, opts) for opts in opt_settings]

        estimator = EthosUPerformanceEstimator(
            self.context,
            self.target,
            self.backends,
        )
        original_metrics, *optimized_metrics = estimate_performance(
            model, estimator, optimizers  # type: ignore
        )

        result = OptimizationPerformanceMetrics(
            original_perf_metrics=original_metrics,
            optimizations_perf_metrics=list(zip(opt_settings, optimized_metrics)),
        )
        return result

    @staticmethod
    def _parse_optimization_params(
        optimizations: list[list[dict]],
    ) -> list[list[OptimizationSettings]]:
        """Parse optimization parameters."""
        if not is_list_of(optimizations, list):
            raise ValueError("Optimization parameters expected to be a list.")

        return [
            [
                OptimizationSettings(
                    item.get("optimization_type"),  # type: ignore
                    item.get("optimization_target"),  # type: ignore
                    item.get("layers_to_optimized"),
                )
                for item in opt_configuration
            ]
            for opt_configuration in optimizations
        ]

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_model_optimizations"
