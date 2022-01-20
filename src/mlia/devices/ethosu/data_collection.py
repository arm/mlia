# Copyright 2021, Arm Ltd.
"""Data collection module for Ethos-U."""
import logging
from pathlib import Path
from typing import List
from typing import Optional

from mlia.core.context import Context
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.performance import estimate_performance
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.performance import EthosUPerformanceEstimator
from mlia.devices.ethosu.performance import OptimizationPerformanceMetrics
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.nn.tensorflow.config import get_keras_model
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.optimizations.select import get_optimizer
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.nn.tensorflow.utils import save_keras_model
from mlia.tools.vela_wrapper import Operators
from mlia.tools.vela_wrapper import supported_operators
from mlia.utils.types import is_list_of

logger = logging.getLogger(__name__)


class EthosUOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path, device: EthosUConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.device = device

    def collect_data(self) -> Operators:
        """Collect operator compatibility information."""
        tflite_model = get_tflite_model(self.model, self.context)

        logger.info("Checking operator compatibility ...")
        ops = supported_operators(
            Path(tflite_model.model_path), self.device.compiler_options
        )
        logger.info("Done\n")
        return ops

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_operator_compatibility"


class EthosUPerformance(ContextAwareDataCollector):
    """Collect performance metrics."""

    def __init__(self, model: Path, device: EthosUConfiguration) -> None:
        """Init performance data collector."""
        self.model = model
        self.device = device

    def collect_data(self) -> PerformanceMetrics:
        """Collect model performance metrics."""
        tflite_model = get_tflite_model(self.model, self.context)
        estimator = EthosUPerformanceEstimator(self.context, self.device)

        return estimator.estimate(tflite_model)

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_performance"


class OptimizeModel:
    """Helper class for model optimization."""

    def __init__(
        self, context: Context, opt_settings: List[OptimizationSettings]
    ) -> None:
        """Init helper."""
        self.context = context
        self.opt_settings = opt_settings

    def __call__(self, keras_model: KerasModel) -> KerasModel:
        """Run optimization."""
        optimizer = get_optimizer(keras_model, self.opt_settings)

        opts_as_str = ", ".join(str(opt) for opt in self.opt_settings)
        logger.info("Applying model optimizations - [%s]", opts_as_str)
        optimizer.apply_optimization()

        model = optimizer.get_model()
        model_path = self.context.get_model_path("optimized_model.h5")
        save_keras_model(model, model_path)

        return KerasModel(model_path)


class EthosUOptimizationPerformance(ContextAwareDataCollector):
    """Collect performance metrics for the optimizations."""

    def __init__(
        self,
        model: Path,
        device: EthosUConfiguration,
        optimizations: List[List[dict]],
    ) -> None:
        """Init performance optimizations data collector."""
        self.model = model
        self.device = device
        self.optimizations = optimizations

    def collect_data(self) -> Optional[OptimizationPerformanceMetrics]:
        """Collect performance metrics for the optimizations."""
        logger.info("Estimate performance ...")

        if not self.optimizations:
            raise FunctionalityNotSupportedError(
                reason="Unable to estimate model optimizations impact",
                description="No optimization targets provided",
            )

        opt_settings = self._parse_optimization_params(self.optimizations)

        try:
            keras_model = get_keras_model(self.model, self.context)
        except NotImplementedError as err:
            raise FunctionalityNotSupportedError(
                reason="Unable to run model optimizations",
                description=f"{self.model} is not a Keras model and "
                "could not be converted to a Keras model",
            ) from err

        optimizers = [OptimizeModel(self.context, opts) for opts in opt_settings]

        estimator = EthosUPerformanceEstimator(self.context, self.device)
        original_metrics, *optimized_metrics = estimate_performance(
            keras_model, estimator, optimizers  # type: ignore
        )

        result = OptimizationPerformanceMetrics(
            original_perf_metrics=original_metrics,
            optimizations_perf_metrics=list(zip(opt_settings, optimized_metrics)),
        )
        return result

    @staticmethod
    def _parse_optimization_params(
        optimizations: List[List[dict]],
    ) -> List[List[OptimizationSettings]]:
        """Parse optimization parameters."""
        if not is_list_of(optimizations, list):
            raise Exception("Optimization parameters expected to be a list")

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
