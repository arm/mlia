# Copyright 2022, Arm Ltd.
"""Event handlers for CLI."""
import logging
from typing import List
from typing import Optional

from mlia.core._typing import OutputFormat
from mlia.core._typing import PathOrFileLike
from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import AdviceEvent
from mlia.core.events import ActionFinishedEvent
from mlia.core.events import ActionStartedEvent
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import ExecutionFailedEvent
from mlia.core.events import SystemEventsHandler
from mlia.devices.ethosu.events import EthosUAdvisorEventHandler
from mlia.devices.ethosu.events import EthosUAdvisorStartedEvent
from mlia.devices.ethosu.performance import OptimizationPerformanceMetrics
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.devices.ethosu.reporters import Reporter
from mlia.tools.vela_wrapper import Operators


logger = logging.getLogger(__name__)


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

    def on_action_started(self, event: ActionStartedEvent) -> None:
        """Handle ActionStarted event."""
        if event.action_type == "applying_optimizations":
            opt_settings = event.params.get("opt_settings") if event.params else ""
            logger.info("Applying optimizations %s ...", opt_settings)

        if event.action_type == "operator_compatibility":
            logger.info("Checking operator compatibility ...")

    def on_action_finished(self, event: ActionFinishedEvent) -> None:
        """Handle ActionFinished event."""
        logger.info("Done")


class ReportingHandler(SystemEventsHandler, EthosUAdvisorEventHandler):
    """Event handler for the reporting."""

    def __init__(
        self,
        output_format: OutputFormat = "plain_text",
        output: Optional[PathOrFileLike] = None,
    ) -> None:
        """Init event handler."""
        self.reporter = Reporter(output_format)
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

    def on_collected_data(self, event: CollectedDataEvent) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, Operators):
            self.reporter.submit([data_item.ops, data_item])

        if isinstance(data_item, PerformanceMetrics):
            self.reporter.submit(data_item)

        if isinstance(data_item, OptimizationPerformanceMetrics):
            original_metrics = data_item.original_perf_metrics
            if not data_item.optimizations_perf_metrics:
                return

            _opt_settings, optimized_metrics = data_item.optimizations_perf_metrics[0]

            self.reporter.submit(
                [original_metrics, optimized_metrics],
                columns_name="Metrics",
                title="Performance metrics",
                space=True,
                notes=(
                    "IMPORTANT: The applied tooling techniques have an impact "
                    "on accuracy. Additional hyperparameter tuning may be required "
                    "after any optimization."
                ),
            )

    def on_advice_event(self, event: AdviceEvent) -> None:
        """Handle Advice event."""
        self.advice.append(event.advice)

    def on_ethos_u_advisor_started(self, event: EthosUAdvisorStartedEvent) -> None:
        """Handle EthosUAdvisorStarted event."""
        self.reporter.submit(event.device)
