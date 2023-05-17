# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Event handler."""
from __future__ import annotations

import logging

from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.target.hydra.events import HydraAdvisorEventHandler
from mlia.target.hydra.events import HydraAdvisorStartedEvent
from mlia.target.hydra.performance import HydraPerformanceMetrics
from mlia.target.hydra.reporters import hydra_formatters

logger = logging.getLogger(__name__)


class HydraEventHandler(WorkflowEventsHandler, HydraAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self) -> None:
        """Init event handler."""
        super().__init__(hydra_formatters)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, HydraPerformanceMetrics):
            self.reporter.submit(data_item, delay_print=True, space=True)

    def on_hydra_advisor_started(self, event: HydraAdvisorStartedEvent) -> None:
        """Handle HydraAdvisorStarted event."""
        self.reporter.submit(event.device)
