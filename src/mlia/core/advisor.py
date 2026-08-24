# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Inference advisor module."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import cast

from mlia.core.common import NamedEntity
from mlia.core.context import Context
from mlia.core.data_analysis import DataAnalyzer, PatternAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.mixins import ParameterResolverMixin
from mlia.core.output_postprocessing import postprocess_standardized_output
from mlia.core.settings import ApplicationSettings
from mlia.core.workflow import DefaultWorkflowExecutor, WorkflowExecutor


class InferenceAdvisor(NamedEntity):
    """Base class for inference advisors."""

    @abstractmethod
    def configure(self, context: Context) -> WorkflowExecutor:
        """Configure advisor execution."""

    def run(
        self, context: Context, settings: ApplicationSettings | None = None
    ) -> dict[str, object] | None:
        """Run the advisor and postprocess its authoritative standardized output."""
        executor = self.configure(context)
        output = executor.run()
        if output is None:
            return None
        return postprocess_standardized_output(
            output, settings or ApplicationSettings()
        )


class DefaultInferenceAdvisor(InferenceAdvisor, ParameterResolverMixin):
    """Default implementation for the advisor."""

    def configure(self, context: Context) -> WorkflowExecutor:
        """Configure advisor."""
        return DefaultWorkflowExecutor(
            context,
            self.get_collectors(context),
            self.get_analyzers(context),
            self.get_pattern_analyzers(context),
        )

    @abstractmethod
    def get_collectors(self, context: Context) -> list[DataCollector]:
        """Return list of the data collectors."""

    @abstractmethod
    def get_analyzers(self, context: Context) -> list[DataAnalyzer]:
        """Return list of the data analyzers."""

    @abstractmethod
    def get_pattern_analyzers(self, context: Context) -> list[PatternAnalyzer]:
        """Return list of the pattern analyzers.

        Pattern analyzers detect patterns in facts and generate composite facts.
        Return empty list if pattern detection is not needed.
        """

    def get_string_parameter(self, context: Context, param: str) -> str:
        """Get string parameter value."""
        value = self.get_parameter(
            self.name(),
            param,
            expected_type=str,
            context=context,
        )

        return cast(str, value)

    def get_model(self, context: Context) -> Path:
        """Get path to the model."""
        model_param = self.get_string_parameter(context, "model")

        model = Path(model_param)
        if not model.exists():
            raise FileNotFoundError(f"Path {model} does not exist.")

        return model

    def get_target_profile(self, context: Context) -> str:
        """Get target profile."""
        return self.get_string_parameter(context, "target_profile")
