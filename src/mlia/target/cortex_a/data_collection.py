# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collection module for Cortex-A."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.core.data_collection import ContextAwareDataCollector
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.tflite_compat import TFLiteChecker
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.target.cortex_a.operators import get_cortex_a_compatibility_info
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


class CortexAOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path) -> None:
        """Init operator compatibility data collector."""
        self.model = model

    def collect_data(self) -> TFLiteCompatibilityInfo | CortexACompatibilityInfo | None:
        """Collect operator compatibility information."""
        if not is_tflite_model(self.model):
            with log_action("Checking TensorFlow Lite compatibility ..."):
                tflite_checker = TFLiteChecker()
                tflite_compat = tflite_checker.check_compatibility(self.model)

            if not tflite_compat.compatible:
                return tflite_compat

        tflite_model = get_tflite_model(self.model, self.context)

        with log_action("Checking operator compatibility ..."):
            return (
                get_cortex_a_compatibility_info(  # pylint: disable=assignment-from-none
                    Path(tflite_model.model_path)
                )
            )

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "cortex_a_operator_compatibility"
