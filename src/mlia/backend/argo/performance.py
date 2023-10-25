# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for Argo performance estimation."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from typing import Callable
from typing import cast
from typing import Union

import docker
from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.config import CONFIG_TO_CLI_OPTION
from mlia.backend.argo.install import DOCKER_IMAGE_NAME
from mlia.backend.errors import BackendExecutionFailed
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.tflite_graph import operator_names_to_types
from mlia.utils.chrometrace import parse_chrometrace
from mlia.utils.logging import log_action
from mlia.utils.proc import Command
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import OutputLogger
from mlia.utils.proc import process_command_output

logger = logging.getLogger(__name__)


ARGO_OUTPUT_DIR = Path("argo-output")


def get_argo_backend_path() -> str | None:
    """Determine whether we should use Docker wrapper or Subprocess to run Argo."""
    mlia_backend_argo_path = os.environ.get("MLIA_BACKEND_ARGO_PATH")
    return mlia_backend_argo_path


def create_argo_command(
    model_path: Path, output_path: Path, argo_cfg: ArgoConfig
) -> list[str]:
    """Return an Argo command."""
    command = [
        str(model_path),
        "--output-dir",
        str(output_path),
    ]

    for name, value in vars(argo_cfg).items():
        if not value:
            continue
        command.extend((CONFIG_TO_CLI_OPTION[name], str(value)))

    return command


def run_argo_in_docker(
    model_path: Path, argo_cfg: ArgoConfig, tmp_dir: Path, output_logger: OutputConsumer
) -> None:
    """Run Argo using Docker wrapper."""
    mounted_model_dir = Path("/models")
    mounted_output_dir = Path("/output")

    command = ["./build-release/src/argo"] + create_argo_command(
        mounted_model_dir / model_path.name, mounted_output_dir, argo_cfg
    )
    volumes = {
        str(model_path.resolve().parent): {
            "bind": str(mounted_model_dir),
            "mode": "ro",
        },
        tmp_dir: {
            "bind": str(mounted_output_dir),
            "mode": "rw",
        },
    }
    client = docker.from_env()  # type: ignore
    logger.debug("Argo command: %s", command)
    output = client.containers.run(
        image=DOCKER_IMAGE_NAME,
        command=command,
        volumes=volumes,
        stdout=True,
        stderr=True,
        remove=True,
    )
    output = cast(bytes, output)
    output_logger(output.decode("utf-8"))


def run_argo_in_subprocess(
    model_path: Path,
    argo_cfg: ArgoConfig,
    tmp_dir: Path,
    output_logger: OutputConsumer,
    argo_path: str,
) -> None:
    """Run Argo using the Subprocess module."""
    command = [argo_path] + create_argo_command(model_path, tmp_dir, argo_cfg)
    logger.debug("Argo command: %s", command)

    try:
        process_command_output(Command(command), [output_logger])
    except subprocess.CalledProcessError as err:
        raise BackendExecutionFailed("Backend execution failed.") from err


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
class ArgoPerformanceMetrics:
    """Collection of Argo configuration and performance metrics."""

    backend_config: ArgoConfig
    metrics_file: Path
    operator_performance_data: list[OperatorPerformanceData]


class ArgoPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], ArgoPerformanceMetrics]
):
    """Performance estimator for Hydra."""

    def __init__(self, output_dir: Path, backend_config: dict) -> None:
        """Init performance estimator."""
        self.backend_config = ArgoConfig(**backend_config.get("argo", {}))
        self.output_dir = output_dir

    def estimate(self, model: Path | ModelConfiguration) -> ArgoPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            metrics_file = self._run_argo(model_path)

            trace = parse_chrometrace(metrics_file)
            op_names_types_map = operator_names_to_types(model_path)

            op_row_durations_map = trace.summarize_durations_per_row(
                # Keep only rows that belong to real rows
                lambda slice: slice.name
                in op_names_types_map
            )
            pass_op_types = _get_pass_types(op_row_durations_map, op_names_types_map)

            op_data = []
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

            return ArgoPerformanceMetrics(self.backend_config, metrics_file, op_data)

    def _run_argo(self, model_path: Path) -> Path:
        """Run Argo and return the path to the Argo stats file."""
        # Use a temporary directory for the output, because docker creates files as
        # root user, which create problems when inside the MLIA output directory.
        # The files are copied from the tmp dir into the MLIA output directory.
        # For the same reason we cannot remove the tmp dir after use.
        tmp_dir = Path(mkdtemp(prefix="mlia-argo-"))

        argo_path = get_argo_backend_path()

        run_argo: Callable[[Path, ArgoConfig, Path, OutputConsumer], None]
        if argo_path:
            run_argo = partial(run_argo_in_subprocess, argo_path=argo_path)
        else:
            run_argo = run_argo_in_docker

        with log_action("Running Argo performance estimation..."):
            output_logger = OutputLogger(logger, logging.INFO)
            run_argo(model_path, self.backend_config, tmp_dir, output_logger)

        with log_action("Verify Argo output files..."):
            argo_output_dir_path = self.output_dir / ARGO_OUTPUT_DIR
            logger.debug(
                "Copying Argo output files from '%s' to '%s'.",
                tmp_dir,
                argo_output_dir_path,
            )
            shutil.rmtree(str(argo_output_dir_path), ignore_errors=True)
            shutil.copytree(str(tmp_dir), str(argo_output_dir_path))
            argo_metrics_file = (
                argo_output_dir_path / f"{model_path.stem}_chrome_trace.json"
            )
            assert argo_metrics_file.is_file(), f"{argo_metrics_file} is not a file!"

        return argo_metrics_file


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
