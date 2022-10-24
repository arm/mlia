# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handler."""
from __future__ import annotations

import logging

from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.core.typing import PathOrFileLike
from mlia.devices.cortexa.events import CortexAAdvisorEventHandler
from mlia.devices.cortexa.events import CortexAAdvisorStartedEvent
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.devices.cortexa.reporters import cortex_a_formatters
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo

logger = logging.getLogger(__name__)


class CortexAEventHandler(WorkflowEventsHandler, CortexAAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self, output: PathOrFileLike | None = None) -> None:
        """Init event handler."""
        super().__init__(cortex_a_formatters, output)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, CortexACompatibilityInfo):
            self.reporter.submit(data_item.operators, delay_print=True)

        if isinstance(data_item, TFLiteCompatibilityInfo) and not data_item.compatible:
            self.reporter.submit(data_item, delay_print=True)

    def on_cortex_a_advisor_started(self, event: CortexAAdvisorStartedEvent) -> None:
        """Handle CortexAAdvisorStarted event."""
        self.reporter.submit(event.device)
