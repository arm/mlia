# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U MLIA module."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from mlia.core._typing import PathOrFileLike
from mlia.core.advice_generation import AdviceProducer
from mlia.core.advisor import DefaultInferenceAdvisor
from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.events import Event
from mlia.devices.ethosu.advice_generation import EthosUAdviceProducer
from mlia.devices.ethosu.advice_generation import EthosUStaticAdviceProducer
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.config import get_target
from mlia.devices.ethosu.data_analysis import EthosUDataAnalyzer
from mlia.devices.ethosu.data_collection import EthosUOperatorCompatibility
from mlia.devices.ethosu.data_collection import EthosUOptimizationPerformance
from mlia.devices.ethosu.data_collection import EthosUPerformance
from mlia.devices.ethosu.events import EthosUAdvisorStartedEvent
from mlia.devices.ethosu.handlers import EthosUEventHandler
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.utils.types import is_list_of


class EthosUInferenceAdvisor(DefaultInferenceAdvisor):
    """Ethos-U Inference Advisor."""

    @classmethod
    def name(cls) -> str:
        """Return name of the advisor."""
        return "ethos_u_inference_advisor"

    def get_collectors(self, context: Context) -> List[DataCollector]:
        """Return list of the data collectors."""
        model = self.get_model(context)
        device = self._get_device(context)
        backends = self._get_backends(context)

        collectors: List[DataCollector] = []

        if AdviceCategory.OPERATORS in context.advice_category:
            collectors.append(EthosUOperatorCompatibility(model, device))

        # Performance and optimization are mutually exclusive.
        # Decide which one to use (taking into account the model format).
        if is_tflite_model(model):
            # TFLite models do not support optimization (only performance)!
            if context.advice_category == AdviceCategory.OPTIMIZATION:
                raise Exception(
                    "Command 'optimization' is not supported for TFLite files."
                )
            if AdviceCategory.PERFORMANCE in context.advice_category:
                collectors.append(EthosUPerformance(model, device, backends))
        else:
            # Keras/SavedModel: Prefer optimization
            if AdviceCategory.OPTIMIZATION in context.advice_category:
                optimization_settings = self._get_optimization_settings(context)
                collectors.append(
                    EthosUOptimizationPerformance(
                        model, device, optimization_settings, backends
                    )
                )
            elif AdviceCategory.PERFORMANCE in context.advice_category:
                collectors.append(EthosUPerformance(model, device, backends))

        return collectors

    def get_analyzers(self, context: Context) -> List[DataAnalyzer]:
        """Return list of the data analyzers."""
        return [
            EthosUDataAnalyzer(),
        ]

    def get_producers(self, context: Context) -> List[AdviceProducer]:
        """Return list of the advice producers."""
        return [
            EthosUAdviceProducer(),
            EthosUStaticAdviceProducer(),
        ]

    def get_events(self, context: Context) -> List[Event]:
        """Return list of the startup events."""
        model = self.get_model(context)
        device = self._get_device(context)

        return [
            EthosUAdvisorStartedEvent(device=device, model=model),
        ]

    def _get_device(self, context: Context) -> EthosUConfiguration:
        """Get device."""
        target_profile = self.get_target_profile(context)

        return get_target(target_profile)

    def _get_optimization_settings(self, context: Context) -> List[List[dict]]:
        """Get optimization settings."""
        return self.get_parameter(  # type: ignore
            EthosUOptimizationPerformance.name(),
            "optimizations",
            expected_type=list,
            expected=False,
            context=context,
        )

    def _get_backends(self, context: Context) -> Optional[List[str]]:
        """Get list of backends."""
        return self.get_parameter(  # type: ignore
            self.name(),
            "backends",
            expected_type=list,
            expected=False,
            context=context,
        )


def configure_and_get_ethosu_advisor(
    context: ExecutionContext,
    target_profile: str,
    model: Union[Path, str],
    output: Optional[PathOrFileLike] = None,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Create and configure Ethos-U advisor."""
    if context.event_handlers is None:
        context.event_handlers = [EthosUEventHandler(output)]

    if context.config_parameters is None:
        context.config_parameters = _get_config_parameters(
            model, target_profile, **extra_args
        )

    return EthosUInferenceAdvisor()


_DEFAULT_OPTIMIZATION_TARGETS = [
    {
        "optimization_type": "pruning",
        "optimization_target": 0.5,
        "layers_to_optimize": None,
    },
    {
        "optimization_type": "clustering",
        "optimization_target": 32,
        "layers_to_optimize": None,
    },
]


def _get_config_parameters(
    model: Union[Path, str],
    target_profile: str,
    **extra_args: Any,
) -> Dict[str, Any]:
    """Get configuration parameters for the advisor."""
    advisor_parameters: Dict[str, Any] = {
        "ethos_u_inference_advisor": {
            "model": model,
            "target_profile": target_profile,
        },
    }

    # Specifying backends is optional (default is used)
    backends = extra_args.get("backends")
    if backends is not None:
        if not is_list_of(backends, str):
            raise Exception("Backends value has wrong format")

        advisor_parameters["ethos_u_inference_advisor"]["backends"] = backends

    optimization_targets = extra_args.get("optimization_targets")
    if not optimization_targets:
        optimization_targets = _DEFAULT_OPTIMIZATION_TARGETS

    if not is_list_of(optimization_targets, dict):
        raise Exception("Optimization targets value has wrong format")

    advisor_parameters.update(
        {
            "ethos_u_model_optimizations": {
                "optimizations": [optimization_targets],
            },
        }
    )

    return advisor_parameters
