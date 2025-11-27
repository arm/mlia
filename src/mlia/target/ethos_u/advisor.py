# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U MLIA module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import cast

from mlia.core.advice_generation import AdviceProducer
from mlia.core.advisor import DefaultInferenceAdvisor
from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.events import Event
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.common.optimization import add_common_optimization_params
from mlia.target.common.optimization import OptimizingDataCollector
from mlia.target.ethos_u.advice_generation import EthosUAdviceProducer
from mlia.target.ethos_u.advice_generation import EthosUStaticAdviceProducer
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.data_analysis import EthosUDataAnalyzer
from mlia.target.ethos_u.data_collection import EthosUOperatorCompatibility
from mlia.target.ethos_u.data_collection import EthosUOptimizationPerformance
from mlia.target.ethos_u.data_collection import EthosUPerformance
from mlia.target.ethos_u.events import EthosUAdvisorStartedEvent
from mlia.target.ethos_u.handlers import EthosUEventHandler
from mlia.target.registry import profile
from mlia.utils.types import is_list_of

logger = logging.getLogger(__name__)


class EthosUInferenceAdvisor(DefaultInferenceAdvisor):
    """Ethos-U Inference Advisor."""

    @classmethod
    def name(cls) -> str:
        """Return name of the advisor."""
        return "ethos_u_inference_advisor"

    def get_collectors(self, context: Context) -> list[DataCollector]:
        """Return list of the data collectors."""
        model = self.get_model(context)
        target_config = self._get_target_config(context)
        target_config.compiler_options.output_dir = context.output_dir  # type: ignore
        backends = self._get_backends(context)

        collectors: list[DataCollector] = []

        if context.category_enabled(AdviceCategory.COMPATIBILITY):
            collectors.append(EthosUOperatorCompatibility(model, target_config))

        # Performance and optimization are mutually exclusive.
        # Decide which one to use (taking into account the model format).
        if is_tflite_model(model):
            # TensorFlow Lite models do not support optimization (only performance)!
            if context.category_enabled(AdviceCategory.OPTIMIZATION):
                optimization_settings = self._get_optimization_settings(context)

                optimization_types = {
                    opt["optimization_type"] for opt in optimization_settings[0]
                }
                if optimization_types != {"rewrite"}:
                    raise RuntimeError(
                        "Only 'rewrite' is supported for TensorFlow Lite files."
                    )

                collectors.append(
                    EthosUOptimizationPerformance(model, target_config, backends)
                )
            if context.category_enabled(AdviceCategory.PERFORMANCE):
                collectors.append(EthosUPerformance(model, target_config, backends))
        else:
            # Keras/SavedModel: Prefer optimization
            if context.category_enabled(AdviceCategory.OPTIMIZATION):
                optimization_settings = self._get_optimization_settings(context)
                collectors.append(
                    EthosUOptimizationPerformance(model, target_config, backends)
                )
            elif context.category_enabled(AdviceCategory.PERFORMANCE):
                collectors.append(EthosUPerformance(model, target_config, backends))

        return collectors

    def get_analyzers(self, context: Context) -> list[DataAnalyzer]:
        """Return list of the data analyzers."""
        return [
            EthosUDataAnalyzer(),
        ]

    def get_producers(self, context: Context) -> list[AdviceProducer]:
        """Return list of the advice producers."""
        return [
            EthosUAdviceProducer(),
            EthosUStaticAdviceProducer(),
        ]

    def get_events(self, context: Context) -> list[Event]:
        """Return list of the startup events."""
        model = self.get_model(context)
        target_config = self._get_target_config(context)

        return [
            EthosUAdvisorStartedEvent(target_config=target_config, model=model),
        ]

    def _get_target_config(self, context: Context) -> EthosUConfiguration:
        """Get target configuration."""
        target_profile = self.get_target_profile(context)
        target_config = cast(EthosUConfiguration, profile(target_profile))
        target_config.compiler_options.output_dir = context.output_dir  # type: ignore
        return target_config

    def _get_optimization_settings(self, context: Context) -> list[list[dict]]:
        """Get optimization settings."""
        return self.get_parameter(  # type: ignore
            OptimizingDataCollector.name(),
            "optimizations",
            expected_type=list,
            expected=False,
            context=context,
        )

    def _get_backends(self, context: Context) -> list[str] | None:
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
    target_profile: str | Path,
    model: str | Path,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Create and configure Ethos-U advisor."""
    if context.event_handlers is None:
        context.event_handlers = [EthosUEventHandler(output_dir=context.output_dir)]

    if context.config_parameters is None:
        context.config_parameters = _get_config_parameters(
            model, target_profile, **extra_args
        )

    return EthosUInferenceAdvisor()


def _get_config_parameters(
    model: str | Path,
    target_profile: str | Path,
    **extra_args: Any,
) -> dict[str, Any]:
    """Get configuration parameters for the advisor."""
    advisor_parameters: dict[str, Any] = {
        "ethos_u_inference_advisor": {
            "model": model,
            "target_profile": target_profile,
        },
    }

    # Specifying backends is optional (default is used)
    backends = extra_args.get("backends")
    if backends is not None:
        if not is_list_of(backends, str):
            raise ValueError("Backends value has wrong format.")

        advisor_parameters["ethos_u_inference_advisor"]["backends"] = backends

    add_common_optimization_params(advisor_parameters, extra_args)

    return advisor_parameters
