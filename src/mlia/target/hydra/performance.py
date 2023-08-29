# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Performance estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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

    def get_percentage_time(self, total_time: float) -> list[list[str]]:
        """Return the percentage of total time taken by each operation."""
        return [
            [str(round((metric["duration"] / total_time) * 100, 2)) + "%"]
            for metric in self.performance
        ]


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
            pass_op_types = _get_pass_types(op_row_durations_map, op_names_types_map)

            for op_name, row_durations_map in op_row_durations_map.items():
                perf = [
                    {
                        "n_pass": tid,
                        "hw_block": trace.get_process(pid).name,
                    }
                    for ((pid, tid), _) in row_durations_map.items()
                ]
                perf[0]["duration"] = _split_pass_times(
                    pass_op_types,
                    trace.get_pass_times(
                        lambda slice: slice.name in op_names_types_map
                    ),
                )[op_name]
                op_type = op_names_types_map.get(op_name, "<unknown>")
                op_data.append(OperatorPerformanceData(op_name, op_type, perf))

            return HydraPerformanceMetrics(self.target_config, metrics_file, op_data)


def _split_pass_times(
    pass_op_types: dict[int, list[dict[str, Any]]],
    pass_times: dict[int, float],
) -> dict[str, float]:
    """
    Return duration of each operator.

    Pass times are split up based on the ranking of the operator type
    Rank A: CE operations
    Rank B: Vector Operations
    Rank C: Other
    """
    op_times = {}
    for pass_number in pass_op_types:
        a_count, b_count, c_count = 0, 0, 0
        for operator in pass_op_types[pass_number]:
            op_type = operator["op_type"]
            if op_type in rank_a:
                a_count += 1
            elif op_type in rank_b:
                b_count += 1
            else:
                c_count += 1

        if a_count > 0:
            a_time = pass_times[pass_number] / a_count
            b_time, c_time = 0.0, 0.0
        elif b_count > 0:
            b_time = pass_times[pass_number] / b_count
            c_time = 0.0
        else:
            c_time = pass_times[pass_number] / c_count

        for operator in pass_op_types[pass_number]:
            if operator["op_type"] in rank_a:
                op_times[operator["op_name"]] = a_time
            elif operator["op_type"] in rank_b:
                op_times[operator["op_name"]] = b_time
            else:
                op_times[operator["op_name"]] = c_time

    return op_times


def _get_pass_types(
    op_row_durations_map: dict[str, dict[tuple[int, int], float]],
    op_names_types_map: dict,
) -> dict[int, list[dict[str, Any]]]:
    """Get all operators and types in each pass."""
    pass_op_types = {}
    for op_name, row_durations_map in op_row_durations_map.items():
        op_type = op_names_types_map.get(op_name, "<unknown>")
        for _, tid in row_durations_map.keys():
            if tid not in pass_op_types:
                pass_op_types[tid] = [
                    {"op_name": op_name, "op_type": op_type.casefold()}
                ]
            else:
                pass_op_types[tid].append(
                    {"op_name": op_name, "op_type": op_type.casefold()}
                )
    return pass_op_types


rank_a = [
    rank.casefold()
    for rank in [
        "AVERAGE_POOL_2D",
        "BATCH_MAT_MUL",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_RNN",
        "CONV_2D",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "DEPTHWISE_CONV_2D",
        "EMBEDDING_LOOKUP",
        "FAKE_QUANT",
        "FULLY_CONNECTED",
        "HASHTABLE_LOOKUP",
        "L2_POOL_2D",
        "LOCAL_RESPONSE_NORMALIZATION",
        "LSH_PROJECTION",
        "MAX_POOL_2D",
        "SKIP_GRAM",
        "SVDF",
        "UNDIRECTIONAL_SEQUENCE_LSTM",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
    ]
]
rank_b = [
    rank.casefold()
    for rank in [
        "ABS",
        "ADD",
        "ADD_N",
        "ARG_MAX",
        "ARG_MIN",
        "ASSIGN_VARIABLE",
        "ATAN2",
        "BIAS_ADD",
        "BROADCAST",
        "BROADCAST_ARGS",
        "BUCKETIZE",
        "CEIL",
        "COMPLEX_ABS",
        "COS",
        "CUMSUM",
        "DEQUANTIZE",
        "DIV",
        "DYNAMIC_UPDATE_SLICE",
        "ELU",
        "EQUAL",
        "EXP",
        "FLOOR",
        "FLOOR_DIV",
        "FLOOR_MOD",
        "GELU",
        "GREATER",
        "GREATER_EQUAL",
        "HARD_SWISH",
        "IMAG",
        "L2_NORMALIZATION",
        "LEAKY_RELU",
        "LESS",
        "LESS_EQUAL",
        "LOG",
        "LOGICAL_AND",
        "LOGICAL_NOT",
        "LOGICAL_OR",
        "LOGISTIC",
        "LOG_SOFTMAX",
        "MAXIMUM",
        "MEAN",
        "MINIMUM",
        "MUL",
        "MULTINOMIAL",
        "NEG",
        "NON_MAX_SUPPRESSION_V4",
        "NON_MAX_SUPPRESSION_V5",
        "NOT_EQUAL",
        "POW",
        "PRELU",
        "RANDOM_STANDARD_NORMAL",
        "RANDOM_UNIFORM",
        "RANGE",
        "READ_VARIABLE",
        "REAL",
        "REDUCE_ALL",
        "REDUCE_ANY",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "REDUCE_PROD",
        "RELU",
        "RELU_0_TO_1",
        "RELU6",
        "RELU_N1_TO_1",
        "RESIZE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "RFFT2D",
        "ROUND",
        "RSQRT",
        "QUANTIZE",
        "SIN",
        "SUB",
        "SQRT",
        "SQUARED_DIFFERENCE",
        "SQUARE",
        "SUM",
        "TANH",
        "UNSORTED_SEGMENT_MAX",
        "UNSORTED_SEGMENT_PROD",
        "UNSORTED_SEGMENT_SUM",
        "VAR_HANDLE",
        "WHERE",
    ]
]
