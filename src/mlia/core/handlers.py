# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handlers module."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import AdviceEvent
from mlia.core.events import ActionFinishedEvent
from mlia.core.events import ActionStartedEvent
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import AnalyzedDataEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.events import DataAnalysisStageFinishedEvent
from mlia.core.events import DataAnalysisStageStartedEvent
from mlia.core.events import DataCollectionStageFinishedEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import DataCollectorSkippedEvent
from mlia.core.events import EventDispatcher
from mlia.core.events import ExecutionFailedEvent
from mlia.core.events import ExecutionFinishedEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.reporting import Report
from mlia.core.reporting import Reporter
from mlia.core.reporting import resolve_output_format
from mlia.core.typing import PathOrFileLike
from mlia.utils.console import create_section_header


logger = logging.getLogger(__name__)


class SystemEventsHandler(EventDispatcher):
    """System events handler."""

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        """Handle ExecutionStarted event."""

    def on_execution_finished(self, event: ExecutionFinishedEvent) -> None:
        """Handle ExecutionFinished event."""

    def on_execution_failed(self, event: ExecutionFailedEvent) -> None:
        """Handle ExecutionFailed event."""

    def on_data_collection_stage_started(
        self, event: DataCollectionStageStartedEvent
    ) -> None:
        """Handle DataCollectionStageStarted event."""

    def on_data_collection_stage_finished(
        self, event: DataCollectionStageFinishedEvent
    ) -> None:
        """Handle DataCollectionStageFinished event."""

    def on_data_collector_skipped(self, event: DataCollectorSkippedEvent) -> None:
        """Handle DataCollectorSkipped event."""

    def on_data_analysis_stage_started(
        self, event: DataAnalysisStageStartedEvent
    ) -> None:
        """Handle DataAnalysisStageStartedEvent event."""

    def on_data_analysis_stage_finished(
        self, event: DataAnalysisStageFinishedEvent
    ) -> None:
        """Handle DataAnalysisStageFinishedEvent event."""

    def on_advice_stage_started(self, event: AdviceStageStartedEvent) -> None:
        """Handle AdviceStageStarted event."""

    def on_advice_stage_finished(self, event: AdviceStageFinishedEvent) -> None:
        """Handle AdviceStageFinished event."""

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedData event."""

    def on_analyzed_data(self, event: AnalyzedDataEvent) -> None:
        """Handle AnalyzedData event."""

    def on_action_started(self, event: ActionStartedEvent) -> None:
        """Handle ActionStarted event."""

    def on_action_finished(self, event: ActionFinishedEvent) -> None:
        """Handle ActionFinished event."""


_ADV_EXECUTION_STARTED = create_section_header("ML Inference Advisor started")
_MODEL_ANALYSIS_MSG = create_section_header("Model Analysis")
_MODEL_ANALYSIS_RESULTS_MSG = create_section_header("Model Analysis Results")
_ADV_GENERATION_MSG = create_section_header("Advice Generation")
_REPORT_GENERATION_MSG = create_section_header("Report Generation")


class WorkflowEventsHandler(SystemEventsHandler):
    """Event handler for the system events."""

    def __init__(
        self,
        formatter_resolver: Callable[[Any], Callable[[Any], Report]],
        output: PathOrFileLike | None = None,
    ) -> None:
        """Init event handler."""
        output_format = resolve_output_format(output)
        self.reporter = Reporter(formatter_resolver, output_format)
        self.output = output

        self.advice: list[Advice] = []

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        """Handle ExecutionStarted event."""
        logger.info(_ADV_EXECUTION_STARTED)

    def on_execution_failed(self, event: ExecutionFailedEvent) -> None:
        """Handle ExecutionFailed event."""
        raise event.err

    def on_data_collection_stage_started(
        self, event: DataCollectionStageStartedEvent
    ) -> None:
        """Handle DataCollectionStageStarted event."""
        logger.info(_MODEL_ANALYSIS_MSG)

    def on_advice_stage_started(self, event: AdviceStageStartedEvent) -> None:
        """Handle AdviceStageStarted event."""
        logger.info(_ADV_GENERATION_MSG)

    def on_data_collector_skipped(self, event: DataCollectorSkippedEvent) -> None:
        """Handle DataCollectorSkipped event."""
        logger.info("Skipped: %s", event.reason)

    @staticmethod
    def report_generated(output: PathOrFileLike) -> None:
        """Log report generation."""
        logger.info(_REPORT_GENERATION_MSG)
        logger.info("Report(s) and advice list saved to: %s", output)

    def on_data_analysis_stage_finished(
        self, event: DataAnalysisStageFinishedEvent
    ) -> None:
        """Handle DataAnalysisStageFinished event."""
        logger.info(_MODEL_ANALYSIS_RESULTS_MSG)

        self.reporter.print_delayed()

    def on_advice_event(self, event: AdviceEvent) -> None:
        """Handle Advice event."""
        self.advice.append(event.advice)

    def on_advice_stage_finished(self, event: AdviceStageFinishedEvent) -> None:
        """Handle AdviceStageFinishedEvent event."""
        self.reporter.submit(
            self.advice,
            show_title=False,
            show_headers=False,
            space="between",
            table_style="no_borders",
        )

        self.reporter.generate_report(self.output)

        if self.output is not None:
            self.report_generated(self.output)
