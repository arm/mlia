# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA Advisor event handlers."""
# pylint: disable=R0801
from __future__ import annotations

import logging

from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.core.typing import PathOrFileLike
from mlia.devices.tosa.events import TOSAAdvisorEventHandler
from mlia.devices.tosa.events import TOSAAdvisorStartedEvent
from mlia.devices.tosa.operators import TOSACompatibilityInfo
from mlia.devices.tosa.reporters import tosa_formatters

logger = logging.getLogger(__name__)


class TOSAEventHandler(WorkflowEventsHandler, TOSAAdvisorEventHandler):
    """Event handler for TOSA advisor."""

    def __init__(self, output: PathOrFileLike | None = None) -> None:
        """Init event handler."""
        super().__init__(tosa_formatters, output)

    def on_tosa_advisor_started(self, event: TOSAAdvisorStartedEvent) -> None:
        """Handle TOSAAdvisorStartedEvent event."""
        self.reporter.submit(event.device)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, TOSACompatibilityInfo):
            self.reporter.submit(data_item.operators, delay_print=True)
