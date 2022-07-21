# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA data collection module."""
import logging
from pathlib import Path

from mlia.core.data_collection import ContextAwareDataCollector
from mlia.devices.tosa.operators import get_tosa_compatibility_info
from mlia.devices.tosa.operators import TOSACompatibilityInfo
from mlia.nn.tensorflow.config import get_tflite_model

logger = logging.getLogger(__name__)


class TOSAOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path) -> None:
        """Init the data collector."""
        self.model = model

    def collect_data(self) -> TOSACompatibilityInfo:
        """Collect TOSA compatibility information."""
        tflite_model = get_tflite_model(self.model, self.context)

        logger.info("Checking operator compatibility ...")
        tosa_info = get_tosa_compatibility_info(tflite_model.model_path)
        logger.info("Done\n")

        return tosa_info

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "tosa_operator_compatibility"
