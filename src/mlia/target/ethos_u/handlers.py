# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handler."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import VelaCompatibilityResult
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.ethos_u.events import EthosUAdvisorEventHandler
from mlia.target.ethos_u.events import EthosUAdvisorStartedEvent
from mlia.target.ethos_u.performance import CombinedPerformanceResult
from mlia.target.ethos_u.performance import CorstonePerformanceResult
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.target.ethos_u.performance import VelaPerformanceResult
from mlia.target.ethos_u.reporters import ethos_u_formatters

logger = logging.getLogger(__name__)


class EthosUEventHandler(WorkflowEventsHandler, EthosUAdvisorEventHandler):
    """CLI event handler."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Init event handler."""
        super().__init__(ethos_u_formatters)
        self.output_dir = output_dir
        self.vela_compatibility_result: VelaCompatibilityResult | None = None
        self.combined_performance_result: CombinedPerformanceResult | None = None
        self.vela_performance_result: VelaPerformanceResult | None = None
        self.corstone_performance_result: CorstonePerformanceResult | None = None

    def on_collected_data(  # pylint: disable=too-many-branches,too-many-statements  # noqa: C901
        self, event: CollectedDataEvent
    ) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, VelaCompatibilityResult):
            # Store for later advice injection
            self.vela_compatibility_result = data_item

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True)

        elif isinstance(data_item, Operators):
            self.reporter.submit([data_item.ops, data_item], delay_print=True)

        if isinstance(data_item, CombinedPerformanceResult):
            # Store for later advice injection
            self.combined_performance_result = data_item

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(data_item, VelaPerformanceResult):
            # Store for later advice injection
            self.vela_performance_result = data_item

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(data_item, CorstonePerformanceResult):
            # Store for later advice injection
            self.corstone_performance_result = data_item

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(data_item, PerformanceMetrics):
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

        if isinstance(data_item, TFLiteCompatibilityInfo) and not data_item.compatible:
            self.reporter.submit(data_item, delay_print=True)

    def on_ethos_u_advisor_started(self, event: EthosUAdvisorStartedEvent) -> None:
        """Handle EthosUAdvisorStarted event."""
        self.reporter.submit(event.target_config)

    def on_advice_stage_finished(self, event: AdviceStageFinishedEvent) -> None:
        """Handle AdviceStageFinished event.

        Write JSON files with advice included in the schema-compliant format.
        """
        # Call parent implementation first
        super().on_advice_stage_finished(event)

        if self.output_dir:
            # Convert advice to schema objects
            schema_advices = [advice.to_schema() for advice in self.advice]

            self._write_json_with_advice(
                self.vela_compatibility_result,
                "vela_compatibility.json",
                schema_advices,
            )
            self._write_json_with_advice(
                self.combined_performance_result,
                "performance.json",
                schema_advices,
            )
            self._write_json_with_advice(
                self.vela_performance_result,
                "vela_performance.json",
                schema_advices,
            )
            self._write_json_with_advice(
                self.corstone_performance_result,
                "corstone_performance.json",
                schema_advices,
            )

    def _write_json_with_advice(
        self,
        result_item: VelaCompatibilityResult
        | CombinedPerformanceResult
        | VelaPerformanceResult
        | CorstonePerformanceResult
        | None,
        filename: str,
        advices: list,
    ) -> None:
        """Write standardized output JSON with advice included.

        Args:
            result_item: Result object with standardized_output
            filename: Output filename
            advices: List of schema Advice objects to include
        """
        if not result_item or not result_item.standardized_output:
            return

        try:
            # Get the standardized output dictionary
            output = result_item.standardized_output

            # Add advice to each result in the output
            if "results" in output:
                for result in output["results"]:
                    result["advices"] = [a.to_dict() for a in advices]

            # Write to file
            output_path = self.output_dir / filename  # type: ignore
            with open(output_path, "w", encoding="utf-8") as file_handle:
                json.dump(output, file_handle, indent=2)
            logger.info("Saved output with advice to %s", output_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to save output to %s: %s", filename, exc)
