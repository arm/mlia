# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA Advisor event handlers."""
# pylint: disable=R0801
from __future__ import annotations

import logging
from pathlib import Path

from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.tosa.data_collection import TOSACompatibilityResult
from mlia.target.tosa.events import TOSAAdvisorEventHandler
from mlia.target.tosa.events import TOSAAdvisorStartedEvent
from mlia.target.tosa.reporters import tosa_formatters

logger = logging.getLogger(__name__)


class TOSAEventHandler(WorkflowEventsHandler, TOSAAdvisorEventHandler):
    """Event handler for TOSA advisor."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Init event handler.

        Args:
            output_dir: Optional output directory for saving standardized outputs
        """
        super().__init__(tosa_formatters)
        self.output_dir = output_dir

    def on_tosa_advisor_started(self, event: TOSAAdvisorStartedEvent) -> None:
        """Handle TOSAAdvisorStartedEvent event."""
        self.reporter.submit(event.target)
        self.reporter.submit(event.tosa_metadata)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        # Handle new TOSACompatibilityResult format
        if isinstance(data_item, TOSACompatibilityResult):
            # Save standardized output if output directory is set
            if self.output_dir and data_item.standardized_output:
                try:
                    # Create output directory if needed
                    self.output_dir.mkdir(parents=True, exist_ok=True)

                    # Generate filename with run_id
                    run_id = data_item.standardized_output.run_id
                    filename = f"tosa_compatibility_{run_id}.json"
                    output_path = self.output_dir / filename

                    # Save the standardized output
                    data_item.standardized_output.save(output_path)
                    logger.info("Saved standardized output to: %s", output_path)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to save standardized output: %s", exc, exc_info=True
                    )

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True)

        # Handle legacy format for backward compatibility
        elif isinstance(data_item, TOSACompatibilityInfo):
            self.reporter.submit(data_item, delay_print=True)

        elif (
            isinstance(data_item, TFLiteCompatibilityInfo) and not data_item.compatible
        ):
            self.reporter.submit(data_item, delay_print=True)
