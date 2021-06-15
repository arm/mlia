# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation."""
from pathlib import Path
from typing import Union

import mlia.tools.vela as vela_wrapper
import numpy as np
from ethosu.vela.architecture_features import ArchitectureFeatures
from ethosu.vela.npu_performance import PassCycles
from numpy.core.records import array


class PerformanceMetrics:
    """Contains all the performance metrics Vela generates in a run."""

    def __init__(
        self,
        npu_cycles: int,
        sram_access_cycles: int,
        dram_access_cycles: int,
        on_chip_flash_access_cycles: int,
        off_chip_flash_access_cycles: int,
        total_cycles: int,
        batch_inference_time: float,
        inferences_per_second: float,
        batch_size: int,
    ) -> None:
        """Initialize the performance metrics instance."""
        self.npu_cycles = npu_cycles
        self.sram_access_cycles = sram_access_cycles
        self.dram_access_cycles = dram_access_cycles
        self.on_chip_flash_access_cycles = on_chip_flash_access_cycles
        self.off_chip_flash_access_cycles = off_chip_flash_access_cycles
        self.total_cycles = total_cycles
        self.batch_inference_time = batch_inference_time
        self.inferences_per_second = inferences_per_second
        self.batch_size = batch_size

        self.cycles_per_batch_unit = "cycles/batch"
        self.inference_time_unit = "ms"
        self.inferences_per_second_unit = "inf/s"


def collect_performance_metrics(model: Union[str, Path]) -> PerformanceMetrics:
    """Collect performance metrics."""
    vela_compiler = vela_wrapper.VelaCompiler()
    optimized_model = vela_compiler.compile_model(model)

    performance_metrics = _get_performance_metrics(
        optimized_model.arch, optimized_model.nng.cycles, optimized_model.nng.batch_size
    )

    return performance_metrics


def _get_performance_metrics(
    arch: ArchitectureFeatures,
    cycles: array,
    batch_size: int,
) -> PerformanceMetrics:
    """Get performace metrics."""
    midpoint_inference_time = cycles[PassCycles.Total] / arch.core_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time
    else:
        midpoint_fps = np.nan

    cycle_counts = {}
    for kind in PassCycles.all():
        aug_label = kind.display_name() + " cycles"
        cyc = cycles[kind]
        cycle_counts[aug_label] = int(cyc)

    performance_metrics = PerformanceMetrics(
        npu_cycles=cycle_counts["NPU cycles"],
        sram_access_cycles=cycle_counts["SRAM Access cycles"],
        dram_access_cycles=cycle_counts["DRAM Access cycles"],
        on_chip_flash_access_cycles=cycle_counts["On-chip Flash Access cycles"],
        off_chip_flash_access_cycles=cycle_counts["Off-chip Flash Access cycles"],
        total_cycles=cycle_counts["Total cycles"],
        batch_inference_time=midpoint_inference_time * 1000,
        inferences_per_second=midpoint_fps,
        batch_size=batch_size,
    )

    return performance_metrics
