# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Performance estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from mlia.backend.argo.performance import estimate_performance
from mlia.core.context import Context
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types
from mlia.target.hydra.config import HydraConfiguration
from mlia.utils.chrometrace import parse_chrometrace
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


@dataclass
class OperatorPerformanceData:
    """Utility container to hold performance report data for one operator/layer."""

    name: str
    op_type: str
    performance: list[dict]

    def get_performance_metrics(self) -> str:
        """Return a report-ready breakdown of the performance report."""
        statreport = ""
        for statitem in self.performance:
            hw_block = statitem["hw_block"]
            duration = statitem["duration"]
            statreport += f"Pass ({hw_block}): {duration} us\n"
        return statreport

    def get_pass_numbers(self) -> list[list[int]]:
        """Return pass numbers."""
        return [[metric["n_pass"]] for metric in self.performance]

    def get_hw_block(self) -> list[list[str]]:
        """Return the name of the HW block."""
        return [[metric["hw_block"]] for metric in self.performance]

    def get_duration(self) -> list[list[str]]:
        """Return the total time spent on an operation.

        Formatting happens here right now - as a workaround for an issue
        found in the reporting library, where it did not handle double
        rows in single cell, or a seamless way to nest tables.
        """
        return [[f'{metric["duration"]:.4f}'] for metric in self.performance]


@dataclass
class HydraPerformanceMetrics:
    """Collection of Hydra configuration and performance metrics."""

    target_config: HydraConfiguration
    metrics_file: Path
    operator_performance_data: list[OperatorPerformanceData]


class HydraPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], HydraPerformanceMetrics]
):
    """Performance estimator for Hydra."""

    def __init__(self, context: Context, target_config: HydraConfiguration) -> None:
        """Init performance estimator."""
        self.context = context
        self.target_config = target_config
        self.output_dir = context.output_dir

    def estimate(self, model: Path | ModelConfiguration) -> HydraPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            metrics_file = estimate_performance(
                model_path,
                self.context.output_dir,
                self.target_config.backend_config,
            )

            op_data = []

            trace = parse_chrometrace(metrics_file)
            op_names_types_map = operator_names_to_types(model_path)

            op_row_durations_map = trace.summarize_durations_per_row(
                # Keep only rows that belong to real rows
                lambda slice: slice.name
                in op_names_types_map
            )

            for op_name, row_durations_map in op_row_durations_map.items():
                perf = [
                    {
                        "n_pass": tid,
                        "hw_block": trace.get_process(pid).name,
                        "duration": duration,
                    }
                    for ((pid, tid), duration) in row_durations_map.items()
                ]

                op_type = op_names_types_map.get(op_name, "<unknown>")
                op_data.append(OperatorPerformanceData(op_name, op_type, perf))

            metrics = HydraPerformanceMetrics(self.target_config, metrics_file, op_data)

            return metrics
