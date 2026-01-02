# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Event handler."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import VelaCompatibilityResult
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

    def on_collected_data(  # pylint: disable=too-many-branches,too-many-statements  # noqa: C901
        self, event: CollectedDataEvent
    ) -> None:
        """Handle CollectedDataEvent event."""
        data_item = event.data_item

        if isinstance(data_item, VelaCompatibilityResult):
            # Save standardized output JSON if available
            if data_item.standardized_output and self.output_dir:
                try:
                    output_path = self.output_dir / "vela_compatibility.json"
                    with open(output_path, "w", encoding="utf-8") as file_handle:
                        json.dump(data_item.standardized_output, file_handle, indent=2)
                    logger.info("Saved Vela compatibility output to %s", output_path)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to save Vela compatibility output: %s", exc)

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True)

        elif isinstance(data_item, Operators):
            self.reporter.submit([data_item.ops, data_item], delay_print=True)

        if isinstance(data_item, CombinedPerformanceResult):
            # Save combined standardized output JSON if available
            if data_item.standardized_output and self.output_dir:
                try:
                    output_path = self.output_dir / "performance.json"
                    with open(output_path, "w", encoding="utf-8") as file_handle:
                        json.dump(data_item.standardized_output, file_handle, indent=2)
                    logger.info("Saved combined performance output to %s", output_path)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "Failed to save combined performance output: %s", exc
                    )

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(data_item, VelaPerformanceResult):
            # Save standardized output JSON if available
            if data_item.standardized_output and self.output_dir:
                try:
                    output_path = self.output_dir / "vela_performance.json"
                    with open(output_path, "w", encoding="utf-8") as file_handle:
                        json.dump(data_item.standardized_output, file_handle, indent=2)
                    logger.info("Saved Vela performance output to %s", output_path)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to save Vela performance output: %s", exc)

            # Submit wrapper object so JSONReporter can access standardized_output
            self.reporter.submit(data_item, delay_print=True, space=True)

        elif isinstance(data_item, CorstonePerformanceResult):
            # Save standardized output JSON if available
            if data_item.standardized_output and self.output_dir:
                try:
                    output_path = self.output_dir / "corstone_performance.json"
                    with open(output_path, "w", encoding="utf-8") as file_handle:
                        json.dump(data_item.standardized_output, file_handle, indent=2)
                    logger.info("Saved standardized output to %s", output_path)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to save standardized output: %s", exc)

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
