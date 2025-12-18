# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handler."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.cortex_a.events import CortexAAdvisorEventHandler
from mlia.target.cortex_a.events import CortexAAdvisorStartedEvent
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.target.cortex_a.operators import CortexACompatibilityResult
from mlia.target.cortex_a.reporters import cortex_a_formatters

logger = logging.getLogger(__name__)


class CortexAEventHandler(WorkflowEventsHandler, CortexAAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Init event handler.

        Args:
            output_dir: Optional output directory for saving standardized outputs
        """
        super().__init__(cortex_a_formatters)
        self.output_dir = output_dir

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, CortexACompatibilityResult):
            # Save standardized output if output directory is set
            if self.output_dir and data_item.standardized_output:
                try:
                    # Create output directory if needed
                    self.output_dir.mkdir(parents=True, exist_ok=True)

                    # Generate filename with run_id
                    run_id = data_item.standardized_output.run_id
                    filename = f"cortex_a_compatibility_{run_id}.json"
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

        elif isinstance(data_item, CortexACompatibilityInfo):
            self.reporter.submit(data_item, delay_print=True)

        if isinstance(data_item, TFLiteCompatibilityInfo) and not data_item.compatible:
            self.reporter.submit(data_item, delay_print=True)

    def on_cortex_a_advisor_started(self, event: CortexAAdvisorStartedEvent) -> None:
        """Handle CortexAAdvisorStarted event."""
        self.reporter.submit(event.target_config)
