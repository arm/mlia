# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA data collection module."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlia.core.output_schema as schema
from mlia.backend.tosa_checker.compat import get_tosa_compatibility_info
from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.tflite_compat import TFLiteChecker
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


@dataclass
class TOSACompatibilityResult:
    """Container for TOSA compatibility results in both formats."""

    legacy_info: TFLiteCompatibilityInfo | TOSACompatibilityInfo | None
    standardized_output: schema.StandardizedOutput | None


class TOSAOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path) -> None:
        """Init the data collector."""
        self.model = model

    def collect_data(
        self,
    ) -> (
        TFLiteCompatibilityInfo | TOSACompatibilityInfo | TOSACompatibilityResult | None
    ):
        """Collect TOSA compatibility information.

        Returns both legacy format and standardized output format.
        """
        # Check TFLite compatibility first if not a TFLite model
        if not is_tflite_model(self.model):
            with log_action("Checking TensorFlow Lite compatibility ..."):
                tflite_checker = TFLiteChecker()
                tflite_compat = tflite_checker.check_compatibility(self.model)

            if not tflite_compat.compatible:
                # Return legacy format for TFLite incompatibility
                # Note: TFLite compatibility could be converted to standardized
                # output in the future if needed
                return tflite_compat

        # Get TFLite model path
        tflite_model = get_tflite_model(self.model, self.context)

        # Collect TOSA compatibility info
        with log_action("Checking operator compatibility ..."):
            tosa_compat = get_tosa_compatibility_info(tflite_model.model_path)

        # Convert to standardized output format using the method on the dataclass
        try:
            # Clean CLI arguments to use basename for executable
            cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []

            # Get TOSA backend configuration
            backend_config = self._get_tosa_backend_config()

            # Use the to_standardized_output method directly
            standardized = tosa_compat.to_standardized_output(
                model_path=self.model,
                cli_arguments=cli_args,
                target_config={"target": "tosa"},
                backend_config=backend_config,
            )

            # Return both formats wrapped in result container
            return TOSACompatibilityResult(
                legacy_info=tosa_compat,
                standardized_output=standardized,
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Log the error but return legacy format to maintain compatibility
            logger.warning(
                "Failed to convert TOSA compatibility to standardized output: %s",
                exc,
                exc_info=True,
            )
            # Return legacy format only
            return tosa_compat

    @staticmethod
    def _get_tosa_backend_config() -> dict[str, Any]:
        """Extract TOSA checker configuration.

        Returns:
            Dictionary with TOSA checker configuration
        """
        config = {}
        try:
            import tosa_checker as tc  # pylint: disable=import-outside-toplevel

            if hasattr(tc, "__version__"):
                config["checker_version"] = tc.__version__
            if hasattr(tc, "TOSA_VERSION"):
                config["tosa_specification_version"] = tc.TOSA_VERSION
        except (ImportError, AttributeError):
            pass
        return config

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "tosa_operator_compatibility"
