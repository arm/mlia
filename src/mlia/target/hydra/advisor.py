# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra advisor module."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mlia.core.advice_generation import AdviceProducer
from mlia.core.advisor import DefaultInferenceAdvisor
from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.errors import ConfigurationError
from mlia.core.events import Event
from mlia.target.common.optimization import add_common_optimization_params
from mlia.target.hydra.advice_generation import HydraAdviceProducer
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.data_analysis import HydraDataAnalyzer
from mlia.target.hydra.data_collection import HydraOptimizingPerformance
from mlia.target.hydra.data_collection import HydraPerformance
from mlia.target.hydra.events import HydraAdvisorStartedEvent
from mlia.target.hydra.handlers import HydraEventHandler


class HydraInferenceAdvisor(DefaultInferenceAdvisor):
    """Hydra Inference Advisor."""

    @classmethod
    def name(cls) -> str:
        """Return name of the advisor."""
        return "hydra_inference_advisor"

    def get_collectors(self, context: Context) -> list[DataCollector]:
        """Return list of the data collectors."""
        model = self.get_model(context)
        target_cfg = self._get_target_cfg(context)

        collectors: list[DataCollector] = []

        backend = self._get_backends(context)[0]

        if context.category_enabled(AdviceCategory.PERFORMANCE):
            collectors.append(HydraPerformance(model, target_cfg, backend))
        elif context.category_enabled(AdviceCategory.OPTIMIZATION):
            collectors.append(HydraOptimizingPerformance(model, target_cfg))
        if context.category_enabled(AdviceCategory.COMPATIBILITY):
            raise ValueError(
                "Only advice category 'PERFORMANCE' and 'OPTIMIZATION' are "
                "currently supported by Hydra."
            )

        return collectors

    def get_analyzers(self, context: Context) -> list[DataAnalyzer]:
        """Return list of the data analyzers."""
        return [
            HydraDataAnalyzer(),
        ]

    def get_producers(self, context: Context) -> list[AdviceProducer]:
        """Return list of the advice producers."""
        return [HydraAdviceProducer()]

    def get_events(self, context: Context) -> list[Event]:
        """Return list of the startup events."""
        model = self.get_model(context)
        target_profile = self.get_target_profile(context)

        return [
            HydraAdvisorStartedEvent(
                model, HydraConfiguration.load_profile(target_profile)
            ),
        ]

    def _get_target_cfg(self, context: Context) -> HydraConfiguration:
        """Get target configuration."""
        target_profile = self.get_target_profile(context)
        return HydraConfiguration.load_profile(target_profile)

    def _get_backends(self, context: Context) -> str:
        """Get list of backends."""
        return self.get_parameter(  # type: ignore
            self.name(),
            "backends",
            expected_type=list,
            expected=True,
            context=context,
        )


def configure_and_get_hydra_advisor(
    context: ExecutionContext,
    target_profile: str | Path,
    model: str | Path,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Create and configure Hydra advisor."""
    if context.event_handlers is None:
        context.event_handlers = [HydraEventHandler()]

    if context.config_parameters is None:
        context.config_parameters = _get_config_parameters(
            model, target_profile, **extra_args
        )

    return HydraInferenceAdvisor()


def _get_config_parameters(
    model: str | Path, target_profile: str | Path, **extra_args: Any
) -> dict[str, Any]:
    """Get configuration parameters for the advisor."""
    advisor_parameters: dict[str, Any] = {
        HydraInferenceAdvisor.name(): {
            "model": str(model),
            "target_profile": target_profile,
        },
    }

    # Hydra requires exactly one backend specified
    backends = extra_args.get("backends")
    if not backends:
        raise ConfigurationError("One backend is required but was not specified.")
    if len(backends) > 1:
        raise ConfigurationError(
            f"Only one backend is supported but {len(backends)} were provided: "
            f"{backends}"
        )
    advisor_parameters[HydraInferenceAdvisor.name()]["backends"] = backends

    add_common_optimization_params(advisor_parameters, extra_args)

    return advisor_parameters
