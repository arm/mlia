# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handler."""
from __future__ import annotations

import logging

from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.cortex_a.events import CortexAAdvisorEventHandler
from mlia.target.cortex_a.events import CortexAAdvisorStartedEvent
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.target.cortex_a.reporters import cortex_a_formatters

logger = logging.getLogger(__name__)


class CortexAEventHandler(WorkflowEventsHandler, CortexAAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self) -> None:
        """Init event handler."""
        super().__init__(cortex_a_formatters)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, CortexACompatibilityInfo):
            self.reporter.submit(data_item.operators, delay_print=True)

        if isinstance(data_item, TFLiteCompatibilityInfo) and not data_item.compatible:
            self.reporter.submit(data_item, delay_print=True)

    def on_cortex_a_advisor_started(self, event: CortexAAdvisorStartedEvent) -> None:
        """Handle CortexAAdvisorStarted event."""
        self.reporter.submit(event.target_config)
