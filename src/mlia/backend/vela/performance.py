# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
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
from pydoc import locate

import numpy as np
from ethosu.vela.npu_performance import PassCycles
from ethosu.vela.tensor import MemArea

from mlia.backend.vela.compiler import OptimizedModel
from mlia.backend.vela.compiler import VelaCompiler
from mlia.backend.vela.compiler import VelaCompilerOptions


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
    unknown_memory_area_size: int
    sram_memory_area_size: int
    dram_memory_area_size: int
    on_chip_flash_memory_area_size: int
    off_chip_flash_memory_area_size: int
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

    def __repr__(self) -> str:
        """Return String Representation of LayerPerfInfo object."""
        header_values = {key: value for key, value, _ in layer_metrics}
        string_to_check = ""
        for field in fields(self):
            string_to_check += (
                f"{header_values[field.name]}: {getattr(self, field.name)}, "
            )
        return string_to_check


@dataclass
class LayerwisePerfInfo:
    """Contains all the per-layer metrics from the per-layer csv file from Vela."""

    layerwise_info: list[LayerPerfInfo]

    def __repr__(self) -> str:
        """Return String Representation of LayerwisePerfInfo object."""
        strings_to_check_layerwise_object = ""
        for layer in self.layerwise_info:
            string_to_check = repr(layer)
            strings_to_check_layerwise_object += string_to_check
        return strings_to_check_layerwise_object


complete_layer_metrics = [
    ("tflite_operator", "TFLite_operator", "TFLite Operator"),
    ("nng_operator", "NNG Operator", "NNG Operator"),
    ("sram_usage", "SRAM Usage", "SRAM Usage"),
    ("peak_percentage", "Peak%", "Peak SRAM Usage (%)"),
    ("op_cycles", "Op Cycles", "OP Cycles"),
    ("network_percentage_1", "Network%", "OP Cycles in Network (%)"),
    ("npu_cycles", "NPU", "NPU Cycles"),
    ("sram_access_cycles", "SRAM AC", "SRAM AC"),
    ("dram_access_cycles", "DRAM AC", "DRAM AC"),
    ("on_chip_flash_access_cycles", "OnFlash AC", "OnFlash AC"),
    ("off_chip_flash_access_cycles", "OffFlash AC", "OffFlash AC"),
    ("mac_count", "MAC Count", "MAC Count"),
    ("network_percentage_2", "Network% (1)", "MAC Count in Network (%)"),
    ("util_mac_percentage", "Util%", "MAC Util (%)"),
    ("name", "Name", "Layer Name"),
]

OUTPUT_METRICS = [field.name for field in fields(LayerPerfInfo)]

layer_metrics = [
    layer_metric
    for layer_metric in complete_layer_metrics
    if layer_metric[0] in OUTPUT_METRICS
]
layer_metrics.sort(key=lambda e: OUTPUT_METRICS.index(e[0]))


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
                key_types = {
                    field.name: locate(str(field.type))
                    for field in fields(LayerPerfInfo)
                }
                ids_to_metrics = {}
                for key, title, _ in metrics:
                    try:
                        ids_to_metrics[key] = key_types[key](  # type: ignore
                            row_as_dict[title]
                        )
                    except ValueError as err:
                        if "invalid literal for int() with base 10" in str(err):
                            ids_to_metrics[key] = key_types[key](  # type: ignore
                                float(row_as_dict[title])
                            )
                        else:
                            raise
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

    initial_model = vela_compiler.read_model(model_path)
    if initial_model.optimized:
        raise ValueError(
            "Unable to estimate performance for the given optimized model."
        )

    optimized_model = vela_compiler.compile_model(initial_model)
    output_dir = optimized_model.compiler_options.output_dir
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

    return _performance_metrics(layerwise_performance_info, optimized_model)


def _performance_metrics(
    layerwise_performance_info: LayerwisePerfInfo, optimized_model: OptimizedModel
) -> PerformanceMetrics:
    """Return performance metrics for optimized model."""
    cycles = optimized_model.nng.cycles

    def memory_usage(mem_area: MemArea) -> int:
        """Get memory usage for the proviced memory area type."""
        memory_used: dict[MemArea, int] = optimized_model.nng.memory_used
        bandwidths = optimized_model.nng.bandwidths

        return memory_used.get(mem_area, 0) if np.sum(bandwidths[mem_area]) > 0 else 0

    midpoint_fps = np.nan
    midpoint_inference_time = cycles[PassCycles.Total] / optimized_model.arch.core_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time

    return PerformanceMetrics(
        npu_cycles=int(cycles[PassCycles.Npu]),
        sram_access_cycles=int(cycles[PassCycles.SramAccess]),
        dram_access_cycles=int(cycles[PassCycles.DramAccess]),
        on_chip_flash_access_cycles=int(cycles[PassCycles.OnChipFlashAccess]),
        off_chip_flash_access_cycles=int(cycles[PassCycles.OffChipFlashAccess]),
        total_cycles=int(cycles[PassCycles.Total]),
        batch_inference_time=midpoint_inference_time * 1000,
        inferences_per_second=midpoint_fps,
        batch_size=optimized_model.nng.batch_size,
        unknown_memory_area_size=memory_usage(MemArea.Unknown),
        sram_memory_area_size=memory_usage(MemArea.Sram),
        dram_memory_area_size=memory_usage(MemArea.Dram),
        on_chip_flash_memory_area_size=memory_usage(MemArea.OnChipFlash),
        off_chip_flash_memory_area_size=memory_usage(MemArea.OffChipFlash),
        layerwise_performance_info=layerwise_performance_info,
    )
