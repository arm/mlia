# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend integration."""
from __future__ import annotations

import base64
import json
import logging
import re
import subprocess  # nosec
from dataclasses import dataclass
from pathlib import Path

from mlia.backend.errors import BackendExecutionFailed
from mlia.backend.repo import get_backend_repository
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.proc import Command
from mlia.utils.proc import process_command_output


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics parsed from generic inference output."""

    npu_active_cycles: int
    npu_idle_cycles: int
    npu_total_cycles: int
    npu_axi0_rd_data_beat_received: int
    npu_axi0_wr_data_beat_written: int
    npu_axi1_rd_data_beat_received: int


class GenericInferenceOutputParser:
    """Generic inference runner output parser."""

    pattern = re.compile(r"<metrics>(.*)</metrics>")

    def __init__(self) -> None:
        """Init parser."""
        self.base64_data: list[str] = []

    def __call__(self, line: str) -> None:
        """Extract base64 strings from the app output."""
        if res_b64 := self.pattern.search(line):
            self.base64_data.append(res_b64.group(1))

    def get_metrics(self) -> PerformanceMetrics:
        """Parse the collected data and return perf metrics."""
        try:
            parsed_metrics = self._parse_data()

            return PerformanceMetrics(
                parsed_metrics["NPU ACTIVE"],
                parsed_metrics["NPU IDLE"],
                parsed_metrics["NPU TOTAL"],
                parsed_metrics["NPU AXI0_RD_DATA_BEAT_RECEIVED"],
                parsed_metrics["NPU AXI0_WR_DATA_BEAT_WRITTEN"],
                parsed_metrics["NPU AXI1_RD_DATA_BEAT_RECEIVED"],
            )
        except Exception as err:
            raise ValueError("Unable to parse output and get metrics.") from err

    def _parse_data(self) -> dict[str, int]:
        """Parse the data."""
        parsed_metrics: dict[str, int] = {}

        for base64_item in self.base64_data:
            res_json = base64.b64decode(base64_item, validate=True)

            for profiling_group in json.loads(res_json):
                for metric in profiling_group["samples"]:
                    metric_name = metric["name"]
                    metric_value = int(metric["value"][0])

                    if metric_name in parsed_metrics:
                        raise KeyError(f"Duplicate key {metric_name}")

                    parsed_metrics[metric_name] = metric_value

        return parsed_metrics


@dataclass
class FVPMetadata:
    """Metadata for FVP."""

    executable: str
    generic_inf_app: Path


def get_generic_inference_app_path(fvp: str, target: str) -> Path:
    """Return path to the generic inference runner binary."""
    apps_path = get_mlia_resources() / "backends/applications"

    fvp_mapping = {"corstone-300": "300", "corstone-310": "310"}
    target_mapping = {"ethos-u55": "U55", "ethos-u65": "U65"}

    fvp_version = f"sse-{fvp_mapping[fvp]}"
    app_version = f"22.08.02-ethos-{target_mapping[target]}-Default-noTA"

    app_dir = f"inference_runner-{fvp_version}-{app_version}"
    return apps_path.joinpath(app_dir, "ethos-u-inference_runner.axf")


def get_executable_name(fvp: str, profile: str, target: str) -> str:
    """Return name of the executable for selected FVP and profile."""
    executable_name_mapping = {
        ("corstone-300", "AVH", "ethos-u55"): "VHT_Corstone_SSE-300_Ethos-U55",
        ("corstone-300", "AVH", "ethos-u65"): "VHT_Corstone_SSE-300_Ethos-U65",
        ("corstone-300", "default", "ethos-u55"): "FVP_Corstone_SSE-300_Ethos-U55",
        ("corstone-300", "default", "ethos-u65"): "FVP_Corstone_SSE-300_Ethos-U65",
        ("corstone-310", "AVH", "ethos-u55"): "VHT_Corstone_SSE-310",
        ("corstone-310", "AVH", "ethos-u65"): "VHT_Corstone_SSE-310_Ethos-U65",
    }

    return executable_name_mapping[(fvp, profile, target)]


def get_fvp_metadata(fvp: str, profile: str, target: str) -> FVPMetadata:
    """Return metadata for selected Corstone backend."""
    executable_name = get_executable_name(fvp, profile, target)
    app = get_generic_inference_app_path(fvp, target)

    return FVPMetadata(executable_name, app)


def build_corstone_command(
    backend_path: Path,
    fvp: str,
    target: str,
    mac: int,
    model: Path,
    profile: str,
) -> Command:
    """Build command to run Corstone FVP."""
    fvp_metadata = get_fvp_metadata(fvp, profile, target)

    cmd = [
        backend_path.joinpath(fvp_metadata.executable).as_posix(),
        "-a",
        fvp_metadata.generic_inf_app.as_posix(),
        "--data",
        f"{model}@0x90000000",
        "-C",
        f"ethosu.num_macs={mac}",
        "-C",
        "mps3_board.telnetterminal0.start_telnet=0",
        "-C",
        "mps3_board.uart0.out_file='-'",
        "-C",
        "mps3_board.uart0.shutdown_on_eot=1",
        "-C",
        "mps3_board.visualisation.disable-visualisation=1",
        "--stat",
    ]

    return Command(cmd)


def get_metrics(
    backend_path: Path,
    fvp: str,
    target: str,
    mac: int,
    model: Path,
    profile: str = "default",
) -> PerformanceMetrics:
    """Run generic inference and return perf metrics."""
    try:
        command = build_corstone_command(
            backend_path,
            fvp,
            target,
            mac,
            model,
            profile,
        )
    except Exception as err:
        raise BackendExecutionFailed(
            f"Unable to construct a command line for {fvp}"
        ) from err

    output_parser = GenericInferenceOutputParser()

    def redirect_to_log(line: str) -> None:
        """Redirect FVP output to the logger."""
        logger.debug(line.strip())

    try:
        process_command_output(
            command,
            [output_parser, redirect_to_log],
        )
    except subprocess.CalledProcessError as err:
        raise BackendExecutionFailed("Backend execution failed.") from err

    return output_parser.get_metrics()


def estimate_performance(
    target: str, mac: int, model: Path, backend: str
) -> PerformanceMetrics:
    """Get performance estimations."""
    backend_repo = get_backend_repository()
    backend_path, settings = backend_repo.get_backend_settings(backend)

    if not settings or "profile" not in settings:
        raise BackendExecutionFailed(f"Unable to configure backend {backend}.")

    return get_metrics(
        backend_path,
        backend,
        target,
        mac,
        model,
        settings["profile"],
    )
