# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Performance estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from mlia.backend.argo.performance import estimate_performance
from mlia.core.context import Context
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.target.hydra.config import HydraConfiguration
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


@dataclass
class HydraPerformanceMetrics:
    """Collection of Hydra configuration and performance metrics."""

    target_config: HydraConfiguration
    metrics_file: Path


class HydraPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], HydraPerformanceMetrics]
):
    """Performance estimator for Hydra."""

    def __init__(self, context: Context, target_config: HydraConfiguration) -> None:
        """Init performance estimator."""
        self.context = context
        self.target_config = target_config
        self.output_dir = context.output_dir

    def estimate(self, model: Path | ModelConfiguration) -> HydraPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            metrics_file = estimate_performance(
                model_path,
                self.context.output_dir,
                self.target_config.backend_config,
            )
            return HydraPerformanceMetrics(self.target_config, metrics_file)
