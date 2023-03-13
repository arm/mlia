# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Performance estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from mlia.core.context import Context
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.target.hydra.config import HydraConfiguration
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


@dataclass
class ArgoStats:
    """Argo stats, including performance metrics."""

    device: HydraConfiguration
    metrics_file: Path


class HydraPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], ArgoStats]
):
    """Argo-based performance estimator."""

    def __init__(self, context: Context, device: HydraConfiguration) -> None:
        """Init performance estimator."""
        self.context = context
        self.device = device

    def estimate(self, model: Path | ModelConfiguration) -> ArgoStats:
        """Estimate performance."""
        with log_action("Getting the Argo performance data..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )
            raise NotImplementedError(
                f"TODO: Implement Argo performance estimation using {model_path}!"
            )
            # argo_stats_file = argo_perf.estimate_performance(model_path, ArgoConfig())
            # return ArgoStats(self.device, argo_stats_file)
