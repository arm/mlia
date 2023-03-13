# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Data collection module for Hydra."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.performance import ArgoStats
from mlia.target.hydra.performance import HydraPerformanceEstimator
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


class HydraPerformance(ContextAwareDataCollector):
    """Collect performance information."""

    def __init__(self, model: Path, cfg: HydraConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.cfg = cfg

    def collect_data(self) -> ArgoStats:
        """Collect operator compatibility information."""
        if not is_tflite_model(self.model):
            raise ConfigurationError("Input must be a tflite file.")

        with log_action("Checking performance..."):
            estimator = HydraPerformanceEstimator(self.context, self.cfg)
            argo_stats = (  # pylint: disable=assignment-from-no-return
                estimator.estimate(self.model)
            )

        return argo_stats

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "hydra_performance"
