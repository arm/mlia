# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Hydra."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceEstimator,
)
from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceMetrics,
)
from mlia.backend.vulkan_model_converter.compat import NGPCompatibilityChecker
from mlia.backend.vulkan_model_converter.compat import NGPModelCompatibilityInfo
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.hydra.config import HydraConfiguration
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


class HydraPerformance(ContextAwareDataCollector):
    """Collect performance information."""

    def __init__(self, model: Path, cfg: HydraConfiguration, backend: str) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg
        self.backend = backend

    def collect_data(
        self,
    ) -> NGPGraphCompilerPerformanceMetrics:
        """Run performance estimator."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        operator_types_mapping = operator_names_to_types(model_path=self.model)
        estimator: NGPGraphCompilerPerformanceEstimator
        if self.backend == "ngp-graph-compiler":
            estimator = NGPGraphCompilerPerformanceEstimator(
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
        return "hydra_performance"


class HydraCompatibility(ContextAwareDataCollector):
    """Collect compatibility information."""

    def __init__(self, model: Path, cfg: HydraConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg

    def collect_data(
        self,
    ) -> NGPModelCompatibilityInfo:
        """Run performance estimator."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        checker = NGPCompatibilityChecker(self.context.output_dir)

        comp_info = checker.check_compatibility(self.model)

        return comp_info

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "hydra_compatibility"
