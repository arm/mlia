# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela performance module."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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

    return _performance_metrics(optimized_model)


def _performance_metrics(optimized_model: OptimizedModel) -> PerformanceMetrics:
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
    )
