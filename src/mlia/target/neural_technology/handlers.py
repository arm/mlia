# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Event handler."""
from __future__ import annotations

import logging

from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceMetrics,
)
from mlia.backend.vulkan_model_converter.compat import NXModelCompatibilityInfo
from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.target.neural_technology.events import NeuralTechnologyAdvisorEventHandler
from mlia.target.neural_technology.events import NeuralTechnologyAdvisorStartedEvent
from mlia.target.neural_technology.reporters import neural_technology_formatters

logger = logging.getLogger(__name__)


class NeuralTechnologyEventHandler(
    WorkflowEventsHandler, NeuralTechnologyAdvisorEventHandler
):
    """CLI event handler."""

    def __init__(self) -> None:
        """Init event handler."""
        super().__init__(neural_technology_formatters)

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(
            data_item,
            (
                NXGraphCompilerPerformanceMetrics,
                NXModelCompatibilityInfo,
            ),
        ):
            self.reporter.submit(data_item, delay_print=True, space=True)

    def on_neural_technology_advisor_started(self, event: NeuralTechnologyAdvisorStartedEvent) -> None:
        """Handle NeuralTechnologyAdvisorStarted event."""
        self.reporter.submit(event.device)
