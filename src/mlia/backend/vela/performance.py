# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela performance module."""
from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from mlia.backend.errors import BackendUnavailableError

try:
    from ethosu.vela import __version__ as ethosu_vela_version

    _VELA_AVAILABLE = True
except ImportError:
    if TYPE_CHECKING:
        from ethosu.vela import __version__ as ethosu_vela_version
    else:

        def __getattr__(name: str) -> Any:
            """Raise BackendUnavailableError for Vela-related attributes."""
            if name in {
                "ethosu_vela_version",
            }:
                raise BackendUnavailableError("Backend vela is not available", "vela")
            raise AttributeError(name)

    _VELA_AVAILABLE = False

import mlia
import mlia.core.output_schema as schema
from mlia.backend.vela.compiler import VelaCompiler
from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.backend.vela.compiler import VelaSummary
from mlia.utils.filesystem import sha256


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:  # pylint: disable=too-many-instance-attributes
    """Contains all the performance metrics Vela generates in a run."""

    npu_cycles: int
    sram_access_cycles: int
    dram_access_cycles: int
    on_chip_flash_access_cycles: int
    off_chip_flash_access_cycles: int
    total_cycles: int
    batch_inference_time: float
    inferences_per_second: float
    batch_size: int
    sram_memory_area_size: float
    dram_memory_area_size: float
    on_chip_flash_memory_area_size: float
    off_chip_flash_memory_area_size: float
    layerwise_performance_info: LayerwisePerfInfo

    def to_standardized_output(  # pylint: disable=too-many-locals
        self,
        model_path: Path,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            target_config: Target configuration dict (target, mac, etc.)
            backend_config: Backend configuration
            run_id: Optional run ID (generated if not provided)
            timestamp: Optional timestamp (generated if not provided)
            cli_arguments: Optional CLI arguments used for the run

        Returns:
            Standardized output dictionary
        """
        # pylint: disable=duplicate-code
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend with version
        try:
            backend_version = ethosu_vela_version
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to get vela version: %s", exc)
            backend_version = "unknown"

        backend = schema.Backend(
            id="vela",
            name="Vela Compiler",
            version=backend_version,
            configuration=backend_config or {},
        )

        # Create target with NPU component
        target_type = (target_config or {}).get("target", "ethos-u")
        mac = (target_config or {}).get("mac", "unknown")

        npu_component = schema.Component(
            type=schema.ComponentType.NPU,
            family=target_type,
            model=None,
            variant=mac if mac != "unknown" else None,
        )

        target = schema.Target(
            profile_name=target_type,
            target_type="npu",
            components=[npu_component],
            configuration=target_config or {},
        )

        # Create model
        model_hash = sha256(model_path)
        model_format = model_path.suffix.lstrip(".") if model_path.suffix else "unknown"
        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
        )

        # Create context
        context = schema.Context(cli_arguments=cli_arguments or [])

        # Create performance metrics
        metrics = [
            schema.Metric(name="npu_cycles", value=self.npu_cycles, unit="cycles"),
            schema.Metric(
                name="sram_access_cycles", value=self.sram_access_cycles, unit="cycles"
            ),
            schema.Metric(
                name="dram_access_cycles", value=self.dram_access_cycles, unit="cycles"
            ),
            schema.Metric(
                name="on_chip_flash_access_cycles",
                value=self.on_chip_flash_access_cycles,
                unit="cycles",
            ),
            schema.Metric(
                name="off_chip_flash_access_cycles",
                value=self.off_chip_flash_access_cycles,
                unit="cycles",
            ),
            schema.Metric(name="total_cycles", value=self.total_cycles, unit="cycles"),
            schema.Metric(
                name="batch_inference_time",
                value=self.batch_inference_time,
                unit="seconds",
            ),
            schema.Metric(
                name="inferences_per_second",
                value=self.inferences_per_second,
                unit="inferences/s",
            ),
            schema.Metric(name="batch_size", value=self.batch_size, unit="count"),
            schema.Metric(
                name="model_size", value=model_path.stat().st_size, unit="bytes"
            ),
            schema.Metric(
                name="sram_memory_area_size",
                value=self.sram_memory_area_size,
                unit="bytes",
            ),
            schema.Metric(
                name="dram_memory_area_size",
                value=self.dram_memory_area_size,
                unit="bytes",
            ),
            schema.Metric(
                name="on_chip_flash_memory_area_size",
                value=self.on_chip_flash_memory_area_size,
                unit="bytes",
            ),
            schema.Metric(
                name="off_chip_flash_memory_area_size",
                value=self.off_chip_flash_memory_area_size,
                unit="bytes",
            ),
        ]

        breakdowns = []
        for layer_info in self.layerwise_performance_info.layerwise_info:
            metrics = [
                schema.Metric(
                    name="op_cycles",
                    value=layer_info.op_cycles,
                    unit="cycles",
                ),
                schema.Metric(
                    name="npu_cycles",
                    value=layer_info.npu_cycles,
                    unit="cycles",
                ),
                schema.Metric(
                    name="sram_access_cycles",
                    value=layer_info.sram_access_cycles,
                    unit="cycles",
                ),
                schema.Metric(
                    name="dram_access_cycles",
                    value=layer_info.dram_access_cycles,
                    unit="cycles",
                ),
                schema.Metric(
                    name="on_chip_flash_access_cycles",
                    value=layer_info.on_chip_flash_access_cycles,
                    unit="cycles",
                ),
                schema.Metric(
                    name="off_chip_flash_access_cycles",
                    value=layer_info.off_chip_flash_access_cycles,
                    unit="cycles",
                ),
                schema.Metric(
                    name="sram_usage",
                    value=layer_info.sram_usage,
                    unit="bytes",
                ),
                schema.Metric(
                    name="mac_count",
                    value=layer_info.mac_count,
                    unit="count",
                ),
                schema.Metric(
                    name="util_mac_percentage",
                    value=layer_info.util_mac_percentage,
                    unit="percent",
                ),
            ]
            breakdowns.append(
                schema.Breakdown(
                    scope=schema.OperatorScope.OPERATOR,
                    name=layer_info.tflite_operator,
                    location=layer_info.name,
                    metrics=metrics,
                )
            )

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer="vela",
            metrics=metrics,
            breakdowns=breakdowns,
        )

        # Build StandardizedOutput
        output = schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
        )

        return output.to_dict()


@dataclass
class LayerPerfInfo:  # pylint: disable=too-many-instance-attributes
    """Contains metrics from a row from the per-layer csv file from Vela."""

    name: str
    tflite_operator: str
    sram_usage: int
    op_cycles: int
    npu_cycles: int
    sram_access_cycles: int
    dram_access_cycles: int
    on_chip_flash_access_cycles: int
    off_chip_flash_access_cycles: int
    mac_count: int
    util_mac_percentage: float


@dataclass
class LayerwisePerfInfo:
    """Contains all the per-layer metrics from the per-layer csv file from Vela."""

    layerwise_info: list[LayerPerfInfo]


complete_layer_metrics = [
    ("tflite_operator", ["TFLite_operator", "Original Operator"], "TFLite Operator"),
    ("nng_operator", "NNG Operator", "NNG Operator"),
    ("sram_usage", ["SRAM Usage", "Staging Usage"], "SRAM Usage"),
    ("peak_percentage", "Peak%", "Peak SRAM Usage (%)"),
    ("op_cycles", "Op Cycles", "OP Cycles"),
    ("network_percentage_1", "Network%", "OP Cycles in Network (%)"),
    ("npu_cycles", "NPU", "NPU Cycles"),
    ("sram_access_cycles", "SRAM AC", "SRAM AC"),
    ("dram_access_cycles", "DRAM AC", "DRAM AC"),
    ("on_chip_flash_access_cycles", "OnFlash AC", "OnFlash AC"),
    ("off_chip_flash_access_cycles", "OffFlash AC", "OffFlash AC"),
    ("mac_count", "MAC Count", "MAC Count"),
    (
        "network_percentage_2",
        ["Network% (1)", "Network% (MAC)"],
        "MAC Count in Network (%)",
    ),
    ("util_mac_percentage", ["Util%", "Util% (MAC)"], "MAC Util (%)"),
    ("name", "Name", "Layer Name"),
]

OUTPUT_METRICS = [field.name for field in fields(LayerPerfInfo)]

layer_metrics = [
    layer_metric
    for layer_metric in complete_layer_metrics
    if layer_metric[0] in OUTPUT_METRICS
]

layer_metrics.sort(key=lambda e: OUTPUT_METRICS.index(e[0]))


def extract_metrics_from_row(row_as_dict: dict, metrics: list, key_types: dict) -> dict:
    """Extract metrics from a CSV row."""
    ids_to_metrics = {}
    for key, title_options, _ in metrics:
        title_found = False
        for title in (
            title_options if isinstance(title_options, list) else [title_options]
        ):
            try:
                ids_to_metrics[key] = key_types[key](row_as_dict[title])
                title_found = True
                break
            except KeyError:
                continue
            except ValueError as err:
                if "invalid literal for int() with base 10" in str(err):
                    ids_to_metrics[key] = key_types[key](float(row_as_dict[title]))
                    title_found = True
                    break
                raise
        if not title_found:
            raise KeyError(f"Title not found for metric key: {key}")
    return ids_to_metrics


def parse_layerwise_perf_csv(  # pylint: disable=too-many-locals
    vela_csv_file: Path, metrics: list
) -> LayerwisePerfInfo:
    """Parse the per-layer csv file from backend vela."""
    if not vela_csv_file.is_file():
        raise FileNotFoundError(f"CSV File not found at {vela_csv_file}\n")
    layerwise_info = []  # type: list[LayerPerfInfo]
    with open(vela_csv_file, encoding="UTF-8") as csv_file:
        layerwise_reader = csv.reader(csv_file, delimiter=",")
        try:
            headers = list(next(layerwise_reader))
        except StopIteration:
            return LayerwisePerfInfo(layerwise_info=layerwise_info)
        headers_to_check_cpu_ops = headers.copy()
        multiple_header_count = Counter(headers)
        # Deal with multiple of the same values in CSV header.
        for idx, header in enumerate(reversed(headers)):
            if multiple_header_count[header] > 1:
                headers[len(headers) - idx - 1] = (
                    headers[len(headers) - idx - 1]
                    + " ("
                    + str(multiple_header_count[header] - 1)
                    + ")"
                )
                multiple_header_count[header] -= 1
        for row in layerwise_reader:
            row_as_dict = dict(zip(headers, row))
            if row == headers_to_check_cpu_ops:
                continue
            try:
                # pylint: disable=eval-used
                key_types = {
                    field.name: eval(field.type)  # type: ignore # nosec
                    for field in fields(LayerPerfInfo)
                }
                # pylint: enable=eval-used
                ids_to_metrics = extract_metrics_from_row(
                    row_as_dict, metrics, key_types
                )
                layerwise_info.append(LayerPerfInfo(**ids_to_metrics))
            except KeyError as err:
                raise KeyError("Generated CSV missing expected headers") from err
    return LayerwisePerfInfo(layerwise_info=layerwise_info)


def estimate_performance(
    model_path: Path, compiler_options: VelaCompilerOptions
) -> PerformanceMetrics:
    """Return performance estimations for the model/target.

    Logic for this function comes from Vela module stats_writer.py
    """
    logger.debug(
        "Estimate performance for the model %s on %s",
        model_path,
        compiler_options.accelerator_config,
    )
    vela_compiler = VelaCompiler(compiler_options)
    if Path(
        Path(compiler_options.output_dir).as_posix()
        + "/"
        + model_path.stem
        + "_summary_"
        + compiler_options.system_config
        + ".csv"
    ).is_file():
        summary_data, _ = vela_compiler.compile_model(model_path, True)
    else:
        summary_data, _ = vela_compiler.compile_model(model_path)

    output_dir = compiler_options.output_dir
    csv_paths = [entry for entry in os.listdir(output_dir) if "per-layer.csv" in entry]
    model_name = str(model_path.stem)
    csv_file_found = None
    for path in csv_paths:
        if model_name in path:
            csv_file_found = path
    if csv_file_found is None:
        raise FileNotFoundError("Vela per-layer CSV file not found")
    csv_path = Path(output_dir) / csv_file_found
    layerwise_performance_info = parse_layerwise_perf_csv(
        vela_csv_file=csv_path, metrics=layer_metrics
    )

    return _performance_metrics(layerwise_performance_info, summary_data)


def _performance_metrics(
    layerwise_performance_info: LayerwisePerfInfo, summary_data: VelaSummary
) -> PerformanceMetrics:
    """Return performance metrics for optimized model."""
    midpoint_fps = 0.0
    midpoint_inference_time = summary_data.cycles_total / summary_data.core_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time

    return PerformanceMetrics(
        npu_cycles=int(summary_data.cycles_npu),
        sram_access_cycles=int(summary_data.cycles_sram_access),
        dram_access_cycles=int(summary_data.cycles_dram_access),
        on_chip_flash_access_cycles=int(summary_data.cycles_on_chip_flash_access),
        off_chip_flash_access_cycles=int(summary_data.cycles_off_chip_flash_access),
        total_cycles=int(summary_data.cycles_total),
        batch_inference_time=midpoint_inference_time * 1000,
        inferences_per_second=midpoint_fps,
        batch_size=summary_data.batch_size,
        sram_memory_area_size=float(summary_data.sram_memory_used),
        dram_memory_area_size=float(summary_data.dram_memory_used),
        on_chip_flash_memory_area_size=float(summary_data.on_chip_flash_memory_used),
        off_chip_flash_memory_area_size=float(summary_data.off_chip_flash_memory_used),
        layerwise_performance_info=layerwise_performance_info,
    )
