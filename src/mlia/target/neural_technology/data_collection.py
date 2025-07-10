# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Neural Technology."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceEstimator,
)
from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceMetrics,
)
from mlia.backend.vulkan_model_converter.compat import NXCompatibilityChecker
from mlia.backend.vulkan_model_converter.compat import NXModelCompatibilityInfo
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


class NeuralTechnologyPerformance(ContextAwareDataCollector):
    """Collect performance information."""

    def __init__(
        self, model: Path, cfg: NeuralTechnologyConfiguration, backend: str
    ) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg
        self.backend = backend

    def collect_data(
        self,
    ) -> NXGraphCompilerPerformanceMetrics:
        """Run performance estimator."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        operator_types_mapping = operator_names_to_types(model_path=self.model)
        estimator: NXGraphCompilerPerformanceEstimator
        if self.backend == "nx-graph-compiler":
            estimator = NXGraphCompilerPerformanceEstimator(
                self.context.output_dir, self.cfg.backend_config, operator_types_mapping
            )
        else:
            raise ValueError(
                f"Backend '{self.backend}' is not supported for "
                f"target '{self.cfg.target}'."
            )

        with log_action("Checking performance..."):
            metrics = estimator.estimate(self.model)

        return metrics

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "neural_technology_performance"


class NeuralTechnologyCompatibility(ContextAwareDataCollector):
    """Collect compatibility information."""

    def __init__(self, model: Path, cfg: NeuralTechnologyConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg

    def collect_data(
        self,
    ) -> NXModelCompatibilityInfo:
        """Run performance estimator."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        checker = NXCompatibilityChecker(self.context.output_dir)

        comp_info = checker.check_compatibility(self.model)

        return comp_info

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "neural_technology_compatibility"
