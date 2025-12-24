# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend integration."""
from __future__ import annotations

import base64
import csv
import json
import logging
import re
import subprocess  # nosec
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import mlia
import mlia.core.output_schema as schema
from mlia.backend.errors import BackendExecutionFailed
from mlia.backend.repo import get_backend_repository
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import sha256
from mlia.utils.proc import Command
from mlia.utils.proc import OutputLogger
from mlia.utils.proc import process_command_output


logger = logging.getLogger(__name__)


# A superset of stats from all corstone versions
_PER_LAYERS_STAT_UNITS = {
    "Staging Usage": "bytes",
    "Peak% (Staging)": "%",
    "Op Cycles": "cycles",
    "Network% (cycles)": "%",
    "NPU": "cycles",
    "SRAM AC": "accesses",
    "DRAM AC": "accesses",
    "OnFlash AC": "accesses",
    "OffFlash AC": "accesses",
    "MAC Count": "operations",
    "Network% (MAC)": "%",
    "Util% (MAC)": "%",
    "SRAM Usage": "bytes",
    "Peak%": "%",
    "Network%": "%",
    "Util%": "%",
}


def _sanitize_metric_name(name: str) -> str:
    name = re.sub(r"[^\w\s]+", "", name).lower()  # Remove non-word-or-space characters
    return name.replace(" ", "_")


def _build_per_layer_metrics(stat: dict) -> list[schema.Metric]:
    metrics = []
    for name, val in stat.items():
        unit = _PER_LAYERS_STAT_UNITS.get(name)
        if unit is None:
            continue
        metrics.append(schema.Metric(_sanitize_metric_name(name), val, unit))
    return metrics


def _parse_per_layer_csv(csv_file: Path) -> list[dict]:
    layer_stats = []
    with open(csv_file, encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            layer_stats.append(dict(row))

    return layer_stats


@dataclass
class CorstoneModelPerformanceMetrics:
    """Model performance metrics."""

    npu_active_cycles: int
    npu_idle_cycles: int
    npu_total_cycles: int
    npu_axi0_rd_data_beat_received: int
    npu_axi0_wr_data_beat_written: int
    npu_axi1_rd_data_beat_received: int
    npu_axi1_wr_data_beat_written: int | None = None

    @classmethod
    def from_fvp_metrics(
        cls,
        target: str,
        fvp_metrics: dict[str, Any],
    ) -> CorstoneModelPerformanceMetrics:
        """Create CorstoneModelPerformanceMetrics from FVP metrics."""
        # Mapping from FVP metric names to class members.
        # Must be in the same order as in the class definition
        target_metric_maps = {
            "default": [
                "NPU ACTIVE",
                "NPU IDLE",
                "NPU TOTAL",
                "NPU AXI0_RD_DATA_BEAT_RECEIVED",
                "NPU AXI0_WR_DATA_BEAT_WRITTEN",
                "NPU AXI1_RD_DATA_BEAT_RECEIVED",
            ],
            "corstone-320": [
                "NPU ACTIVE",
                "NPU IDLE",
                "NPU TOTAL",
                "NPU ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
                "NPU ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN",
                "NPU ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED",
                "NPU ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN",
            ],
        }
        metric_names = target_metric_maps.get(target, target_metric_maps["default"])
        class_fields = list(
            cls.__dataclass_fields__.keys()  # pylint: disable=no-member
        )
        class_kwargs = {}
        for idx, metric_name in enumerate(metric_names):
            if metric_name in fvp_metrics and idx < len(class_fields):
                class_kwargs[class_fields[idx]] = fvp_metrics[metric_name]
            else:
                raise KeyError(f"Metric {metric_name} not found in parsed data.")
        return cls(**class_kwargs)


@dataclass
class CorstonePerformanceMetrics:
    """Performance metrics parsed from generic inference output."""

    npu_model_stats: CorstoneModelPerformanceMetrics
    npu_per_layer_stats: list = field(default_factory=list)

    @classmethod
    def from_fvp_out(
        cls, target: str, metrics: dict[str, Any], per_layer_file: Path
    ) -> CorstonePerformanceMetrics:
        """Create CorstoneModelPerformanceMetrics from FVP output."""
        return cls(
            CorstoneModelPerformanceMetrics.from_fvp_metrics(target, metrics),
            _parse_per_layer_csv(per_layer_file),
        )

    def to_standardized_output(  # pylint: disable=too-many-locals
        self,
        model_path: Path,
        backend_name: str,
        target_config: dict[str, Any],
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> Any:  # Returns StandardizedOutput but avoid circular import
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            backend_name: Name of the Corstone backend (e.g., "corstone-300")
            target_config: Target configuration parameters
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            backend_config: Optional backend configuration parameters

        Returns:
            StandardizedOutput object with performance results
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend
        backend = schema.Backend(
            id=backend_name,
            name=f"Corstone {backend_name.split('-')[1]}"
            if "-" in backend_name
            else backend_name,
            version="unknown",  # Corstone version comes from FVP executable
            configuration=backend_config or {},
        )

        # Extract target info from config
        target_type = target_config.get("target_type", backend_name)
        profile_name = target_config.get("profile_name", target_type)
        mac_config = target_config.get("mac", "unknown")

        # Create target components - only NPU
        components = []

        # Add NPU component if target info available
        npu_target = target_config.get("npu_target") or target_config.get("target")
        if npu_target:
            npu_family = npu_target.split("-")[0] if "-" in npu_target else "ethos-u"
            npu_model = npu_target.split("-")[1] if "-" in npu_target else None

            components.append(
                schema.Component(
                    type=schema.ComponentType.NPU,
                    family=npu_family,
                    model=npu_model,
                    variant=str(mac_config) if mac_config != "unknown" else None,
                )
            )

        target = schema.Target(
            profile_name=profile_name,
            target_type=target_type,
            components=components,
            configuration=target_config,
            description=f"Corstone {backend_name} FVP simulation",
        )

        # Create model info
        model_hash = sha256(model_path)
        model_size = model_path.stat().st_size if model_path.exists() else None
        suffix = model_path.suffix.lower()
        if suffix == ".tflite":
            model_format = "tflite"
        elif suffix in [".vela", ".tflite.vela"]:
            model_format = "vela"
        else:
            model_format = suffix.lstrip(".") or "unknown"

        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
            size_bytes=model_size,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
            runtime_configuration=None,
            git=None,
            notes=None,
        )

        # Create performance metrics
        metrics = [
            schema.Metric(
                name="npu_active_cycles",
                value=float(self.npu_model_stats.npu_active_cycles),
                unit="cycles",
            ),
            schema.Metric(
                name="npu_idle_cycles",
                value=float(self.npu_model_stats.npu_idle_cycles),
                unit="cycles",
            ),
            schema.Metric(
                name="npu_total_cycles",
                value=float(self.npu_model_stats.npu_total_cycles),
                unit="cycles",
            ),
            schema.Metric(
                name="npu_axi0_rd_data_beat_received",
                value=float(self.npu_model_stats.npu_axi0_rd_data_beat_received),
                unit="beats",
            ),
            schema.Metric(
                name="npu_axi0_wr_data_beat_written",
                value=float(self.npu_model_stats.npu_axi0_wr_data_beat_written),
                unit="beats",
            ),
            schema.Metric(
                name="npu_axi1_rd_data_beat_received",
                value=float(self.npu_model_stats.npu_axi1_rd_data_beat_received),
                unit="beats",
            ),
        ]

        breakdowns = []
        for stat in self.npu_per_layer_stats:
            breakdowns.append(
                schema.Breakdown(
                    scope=schema.OperatorScope.OPERATOR,
                    name=stat["NNG Operator"],
                    location=stat["Name"],
                    metrics=_build_per_layer_metrics(stat),
                )
            )

        if self.npu_model_stats.npu_axi1_wr_data_beat_written is not None:
            metrics.append(
                schema.Metric(
                    name="npu_axi1_wr_data_beat_written",
                    value=float(self.npu_model_stats.npu_axi1_wr_data_beat_written),
                    unit="beats",
                )
            )

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer=backend.id,
            warnings=[],
            errors=[],
            metrics=metrics,
            mode=schema.ModeType.SIMULATED,  # Corstone is simulation
            breakdowns=breakdowns,
        )

        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        ).to_dict()


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

    def get_metrics(
        self, output_dir: Path, target: str = "default"
    ) -> CorstonePerformanceMetrics:
        """Parse the collected data and return perf metrics."""
        try:
            parsed_metrics = self._parse_data()
            return CorstonePerformanceMetrics.from_fvp_out(
                target, parsed_metrics, list(output_dir.glob("*_per-layer.csv"))[0]
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

    fvp_mapping = {"corstone-300": "300", "corstone-310": "310", "corstone-320": "320"}
    target_mapping = {"ethos-u55": "U55", "ethos-u65": "U65", "ethos-u85": "U85"}

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
        ("corstone-310", "default", "ethos-u55"): "FVP_Corstone_SSE-310",
        ("corstone-310", "default", "ethos-u65"): "FVP_Corstone_SSE-310_Ethos-U65",
        ("corstone-320", "AVH", "ethos-u85"): "VHT_Corstone_SSE-320",
        ("corstone-320", "default", "ethos-u85"): "FVP_Corstone_SSE-320",
    }

    return executable_name_mapping[(fvp, profile, target)]


def get_fvp_metadata(fvp: str, profile: str, target: str) -> FVPMetadata:
    """Return metadata for selected Corstone backend."""
    executable_name = get_executable_name(fvp, profile, target)

    app = get_generic_inference_app_path(fvp, target)

    return FVPMetadata(executable_name, app)


@dataclass
class CorstoneRunConfig:
    """Configuration for running Corstone FVP generic inference."""

    output_dir: Path
    backend_path: Path
    fvp: str
    target: str
    mac: int
    model: Path
    profile: str = "default"


def build_corstone_command(cfg: CorstoneRunConfig) -> Command:
    """Build command to run Corstone FVP."""
    fvp_metadata = get_fvp_metadata(cfg.fvp, cfg.profile, cfg.target)

    if cfg.fvp == "corstone-320":
        cmd = [
            cfg.backend_path.joinpath(fvp_metadata.executable).as_posix(),
            "-a",
            fvp_metadata.generic_inf_app.as_posix(),
            "--data",
            f"{cfg.model}@0x90000000",
            "-C",
            f"mps4_board.subsystem.ethosu.num_macs={cfg.mac}",
            "-C",
            "mps4_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps4_board.uart0.out_file='-'",
            "-C",
            "mps4_board.uart0.shutdown_on_eot=1",
            "-C",
            "mps4_board.visualisation.disable-visualisation=1",
            "-C",
            "vis_hdlcd.disable_visualisation=1",
            "--stat",
        ]
    else:
        cmd = [
            cfg.backend_path.joinpath(fvp_metadata.executable).as_posix(),
            "-a",
            fvp_metadata.generic_inf_app.as_posix(),
            "--data",
            f"{cfg.model}@0x90000000",
            "-C",
            f"ethosu.num_macs={cfg.mac}",
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


def get_metrics(cfg: CorstoneRunConfig) -> CorstonePerformanceMetrics:
    """Run generic inference and return perf metrics."""
    try:
        command = build_corstone_command(cfg)
    except Exception as err:  # noqa: BLE001 - we want to wrap any construction errors
        raise BackendExecutionFailed(
            f"Unable to construct a command line for {cfg.fvp}"
        ) from err

    output_parser = GenericInferenceOutputParser()
    output_logger = OutputLogger(logger)

    try:
        process_command_output(
            command,
            [output_parser, output_logger],
        )
    except subprocess.CalledProcessError as err:
        raise BackendExecutionFailed("Backend execution failed.") from err

    return output_parser.get_metrics(cfg.output_dir, cfg.fvp)


def estimate_performance(
    target: str, mac: int, model: Path, backend: str, output_dir: Path
) -> CorstonePerformanceMetrics:
    """Get performance estimations."""
    backend_repo = get_backend_repository()
    backend_path, settings = backend_repo.get_backend_settings(backend)

    if not settings or "profile" not in settings:
        raise BackendExecutionFailed(f"Unable to configure backend {backend}.")

    cfg = CorstoneRunConfig(
        output_dir=output_dir,
        backend_path=backend_path,
        fvp=backend,
        target=target,
        mac=mac,
        model=model,
        profile=settings["profile"],
    )
    return get_metrics(cfg)
