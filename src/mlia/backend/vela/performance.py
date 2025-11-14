# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
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

import numpy as np

from mlia.backend.vela.compiler import VelaCompiler
from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.backend.vela.compiler import VelaSummary


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
    midpoint_fps = np.nan
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
