# Copyright 2021, Arm Ltd.
"""Ethos-U IA module."""
from pathlib import Path
from typing import List

from mlia.core.advice_generation import AdviceProducer
from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.mixins import ParameterResolverMixin
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.core.workflow import WorkflowExecutor
from mlia.devices.ethosu.advice_generation import EthosUAdviceProducer
from mlia.devices.ethosu.advice_generation import EthosUStaticAdviceProducer
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.config import get_target
from mlia.devices.ethosu.data_analysis import EthosUDataAnalyzer
from mlia.devices.ethosu.data_collection import EthosUOperatorCompatibility
from mlia.devices.ethosu.data_collection import EthosUOptimizationPerformance
from mlia.devices.ethosu.data_collection import EthosUPerformance
from mlia.devices.ethosu.events import EthosUAdvisorStartedEvent


class EthosUInferenceAdvisor(InferenceAdvisor, ParameterResolverMixin):
    """Ethos-U Inference Advisor."""

    @classmethod
    def name(cls) -> str:
        """Return name of the advisor."""
        return "ethos_u_inference_advisor"

    def configure(self, context: Context) -> WorkflowExecutor:
        """Configure advisor execution."""
        model = self._get_model(context)
        device = self._get_device(context)

        collectors = self._get_collectors(context, model, device)
        analyzers = self._get_analyzers()
        producers = self._get_advice_producers()

        return DefaultWorkflowExecutor(
            context,
            collectors,
            analyzers,
            producers,
            before_start_events=[
                EthosUAdvisorStartedEvent(device=device, model=model),
            ],
        )

    def _get_collectors(
        self, context: Context, model: Path, device: EthosUConfiguration
    ) -> List[DataCollector]:
        """Get collectors."""
        collectors: List[DataCollector] = []

        if context.any_category_enabled(
            AdviceCategory.OPERATORS,
            AdviceCategory.ALL,
        ):
            collectors.append(EthosUOperatorCompatibility(model, device))

        if context.category_enabled(AdviceCategory.PERFORMANCE):
            collectors.append(EthosUPerformance(model, device))

        if context.any_category_enabled(
            AdviceCategory.OPTIMIZATION,
            AdviceCategory.ALL,
        ):
            optimization_settings = self._get_optimization_settings(context)
            collectors.append(
                EthosUOptimizationPerformance(model, device, optimization_settings)
            )

        return collectors

    @staticmethod
    def _get_analyzers() -> List[DataAnalyzer]:
        """Return data analyzers."""
        return [
            EthosUDataAnalyzer(),
        ]

    @staticmethod
    def _get_advice_producers() -> List[AdviceProducer]:
        """Return advice producers."""
        return [
            EthosUAdviceProducer(),
            EthosUStaticAdviceProducer(),
        ]

    def _get_device(self, context: Context) -> EthosUConfiguration:
        """Get device."""
        device_params = self.get_parameter(
            self.name(),
            "device",
            expected_type=dict,
            context=context,
        )

        if "target" not in device_params:
            raise Exception("Unable to get device details")

        return get_target(**device_params)

    def _get_model(self, context: Context) -> Path:
        """Get path to the model."""
        model_param = self.get_parameter(
            self.name(),
            "model",
            expected_type=str,
            context=context,
        )

        if not (model := Path(model_param)).exists():
            raise Exception(f"Path {model} does not exist")

        return model

    def _get_optimization_settings(self, context: Context) -> List[List[dict]]:
        """Get optimization settings."""
        return self.get_parameter(  # type: ignore
            EthosUOptimizationPerformance.name(),
            "optimizations",
            expected_type=list,
            expected=False,
            context=context,
        )
