# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collection module for Cortex-A."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.core.data_collection import ContextAwareDataCollector
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.devices.cortexa.operators import get_cortex_a_compatibility_info
from mlia.nn.tensorflow.config import get_tflite_model

logger = logging.getLogger(__name__)


class CortexAOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path) -> None:
        """Init operator compatibility data collector."""
        self.model = model

    def collect_data(self) -> CortexACompatibilityInfo:
        """Collect operator compatibility information."""
        tflite_model = get_tflite_model(self.model, self.context)

        logger.info("Checking operator compatibility ...")
        ops = get_cortex_a_compatibility_info(Path(tflite_model.model_path))
        logger.info("Done\n")
        return ops

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "cortex_a_operator_compatibility"
