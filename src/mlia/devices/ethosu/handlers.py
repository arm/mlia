# Copyright (C) 2021-2022, Arm Ltd.
"""Event handler."""
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from mlia.core._typing import OutputFormat
from mlia.core._typing import PathOrFileLike
from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import AdviceEvent
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.events import DataAnalysisStageFinishedEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import DataCollectorSkippedEvent
from mlia.core.events import ExecutionFailedEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.events import SystemEventsHandler
from mlia.core.reporting import Reporter
from mlia.devices.ethosu.events import EthosUAdvisorEventHandler
from mlia.devices.ethosu.events import EthosUAdvisorStartedEvent
from mlia.devices.ethosu.performance import OptimizationPerformanceMetrics
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.devices.ethosu.reporters import find_appropriate_formatter
from mlia.tools.vela_wrapper import Operators

logger = logging.getLogger(__name__)

ADV_EXECUTION_STARTED = """
=== Inference advisor started ==============================================
"""

MODEL_ANALYSIS_MSG = """
=== Model Analysis =========================================================
"""

ADV_GENERATION_MSG = """
=== Advice Generation ======================================================
"""

REPORT_GENERATION_MSG = """
=== Report Generation ======================================================
"""


class WorkflowEventsHandler(SystemEventsHandler):
    """Event handler for the system events."""

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        """Handle ExecutionStarted event."""
        logger.info(ADV_EXECUTION_STARTED)

    def on_execution_failed(self, event: ExecutionFailedEvent) -> None:
        """Handle ExecutionFailed event."""
        raise event.err

    def on_data_collection_stage_started(
        self, event: DataCollectionStageStartedEvent
    ) -> None:
        """Handle DataCollectionStageStarted event."""
        logger.info(MODEL_ANALYSIS_MSG)

    def on_advice_stage_started(self, event: AdviceStageStartedEvent) -> None:
        """Handle AdviceStageStarted event."""
        logger.info(ADV_GENERATION_MSG)

    def on_data_collector_skipped(self, event: DataCollectorSkippedEvent) -> None:
        """Handle DataCollectorSkipped event."""
        logger.info("Skipped: %s", event.reason)


class EthosUEventHandler(WorkflowEventsHandler, EthosUAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self, output: Optional[PathOrFileLike] = None) -> None:
        """Init event handler."""
        output_format = self.resolve_output_format(output)

        self.reporter = Reporter(find_appropriate_formatter, output_format)
        self.output = output
        self.advice: List[Advice] = []

    def on_advice_stage_finished(self, event: AdviceStageFinishedEvent) -> None:
        """Handle AdviceStageFinishedEvent event."""
        self.reporter.submit(
            self.advice,
            show_title=False,
            show_headers=False,
            space="between",
            tablefmt="plain",
        )

        self.reporter.generate_report(self.output)

        if self.output is not None:
            logger.info(REPORT_GENERATION_MSG)
            logger.info("Report(s) and advice list saved to: %s", self.output)

    def on_data_analysis_stage_finished(
        self, event: DataAnalysisStageFinishedEvent
    ) -> None:
        """Handle DataAnalysisStageFinished event."""
        self.reporter.print_delayed()

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, Operators):
            self.reporter.submit(
                [data_item.ops, data_item],
                delay_print=True,
                space="top",
            )

        if isinstance(data_item, PerformanceMetrics):
            self.reporter.submit(data_item, delay_print=True)

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

    def on_advice_event(self, event: AdviceEvent) -> None:
        """Handle Advice event."""
        self.advice.append(event.advice)

    def on_ethos_u_advisor_started(self, event: EthosUAdvisorStartedEvent) -> None:
        """Handle EthosUAdvisorStarted event."""
        self.reporter.submit(event.device)

    @staticmethod
    def resolve_output_format(output: Optional[PathOrFileLike]) -> OutputFormat:
        """Resolve output format based on the output name."""
        output_format: OutputFormat = "plain_text"

        if isinstance(output, str):
            output_path = Path(output)
            output_formats: Dict[str, OutputFormat] = {
                ".csv": "csv",
                ".json": "json",
                ".txt": "plain_text",
            }

            if (suffix := output_path.suffix) in output_formats:
                return output_formats[suffix]

        return output_format
