# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handler."""
from __future__ import annotations

import logging

from mlia.backend.vela.compat import Operators
from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.target.ethos_u.events import EthosUAdvisorEventHandler
from mlia.target.ethos_u.events import EthosUAdvisorStartedEvent
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.target.ethos_u.reporters import ethos_u_formatters

logger = logging.getLogger(__name__)


class EthosUEventHandler(WorkflowEventsHandler, EthosUAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self) -> None:
        """Init event handler."""
        super().__init__(ethos_u_formatters)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, Operators):
            self.reporter.submit([data_item.ops, data_item], delay_print=True)

        if isinstance(data_item, PerformanceMetrics):
            self.reporter.submit(data_item, delay_print=True, space=True)

        if isinstance(data_item, OptimizationPerformanceMetrics):
            original_metrics = data_item.original_perf_metrics
            if not data_item.optimizations_perf_metrics:
                return

            _opt_settings, optimized_metrics = data_item.optimizations_perf_metrics[0]

            self.reporter.submit(
                [original_metrics, optimized_metrics],
                delay_print=True,
                columns_name="Metrics",
                title="Performance metrics",
                space=True,
            )

    def on_ethos_u_advisor_started(self, event: EthosUAdvisorStartedEvent) -> None:
        """Handle EthosUAdvisorStarted event."""
        self.reporter.submit(event.target)
