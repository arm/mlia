# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA data collection module."""
from pathlib import Path

from mlia.backend.tosa_checker.compat import get_tosa_compatibility_info
from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.utils.logging import log_action


class TOSAOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path) -> None:
        """Init the data collector."""
        self.model = model

    def collect_data(self) -> TOSACompatibilityInfo:
        """Collect TOSA compatibility information."""
        tflite_model = get_tflite_model(self.model, self.context)

        with log_action("Checking operator compatibility ..."):
            return get_tosa_compatibility_info(tflite_model.model_path)

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "tosa_operator_compatibility"
