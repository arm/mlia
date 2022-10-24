# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import mlia.backend.manager as backend_manager
import mlia.tools.vela_wrapper as vela
from mlia.core.context import Context
from mlia.core.performance import PerformanceEstimator
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.utils.logging import log_action


logger = logging.getLogger(__name__)


@dataclass
class NPUCycles:
    """NPU cycles metrics."""

    npu_active_cycles: int
    npu_idle_cycles: int
    npu_total_cycles: int
    npu_axi0_rd_data_beat_received: int
    npu_axi0_wr_data_beat_written: int
    npu_axi1_rd_data_beat_received: int


BYTES_PER_KILOBYTE = 1024


class MemorySizeType(Enum):
    """Memory size type enumeration."""

    BYTES = 0
    KILOBYTES = 1


@dataclass
class MemoryUsage:
    """Memory usage metrics."""

    sram_memory_area_size: int | float
    dram_memory_area_size: int | float
    unknown_memory_area_size: int | float
    on_chip_flash_memory_area_size: int | float
    off_chip_flash_memory_area_size: int | float
    memory_size_type: MemorySizeType = MemorySizeType.BYTES

    _default_columns = [
        "SRAM used",
        "DRAM used",
        "Unknown memory used",
        "On chip flash used",
        "Off chip flash used",
    ]

    def in_kilobytes(self) -> MemoryUsage:
        """Return memory usage with values in kilobytes."""
        if self.memory_size_type == MemorySizeType.KILOBYTES:
            return self

        kilobytes = [
            value / BYTES_PER_KILOBYTE
            for value in [
                self.sram_memory_area_size,
                self.dram_memory_area_size,
                self.unknown_memory_area_size,
                self.on_chip_flash_memory_area_size,
                self.off_chip_flash_memory_area_size,
            ]
        ]

        return MemoryUsage(
            *kilobytes,  # type: ignore
            memory_size_type=MemorySizeType.KILOBYTES,
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    device: EthosUConfiguration
    npu_cycles: NPUCycles | None
    memory_usage: MemoryUsage | None

    def in_kilobytes(self) -> PerformanceMetrics:
        """Return metrics with memory usage in KiB."""
        if self.memory_usage is None:
            return PerformanceMetrics(self.device, self.npu_cycles, self.memory_usage)

        return PerformanceMetrics(
            self.device, self.npu_cycles, self.memory_usage.in_kilobytes()
        )


@dataclass
class OptimizationPerformanceMetrics:
    """Optimization performance metrics."""

    original_perf_metrics: PerformanceMetrics
    optimizations_perf_metrics: list[
        tuple[list[OptimizationSettings], PerformanceMetrics]
    ]


class VelaPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], MemoryUsage]
):
    """Vela based performance estimator."""

    def __init__(self, context: Context, device: EthosUConfiguration) -> None:
        """Init Vela based performance estimator."""
        self.context = context
        self.device = device

    def estimate(self, model: Path | ModelConfiguration) -> MemoryUsage:
        """Estimate performance."""
        with log_action("Getting the memory usage metrics ..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            vela_perf_metrics = vela.estimate_performance(
                model_path, self.device.compiler_options
            )

            return MemoryUsage(
                vela_perf_metrics.sram_memory_area_size,
                vela_perf_metrics.dram_memory_area_size,
                vela_perf_metrics.unknown_memory_area_size,
                vela_perf_metrics.on_chip_flash_memory_area_size,
                vela_perf_metrics.off_chip_flash_memory_area_size,
            )


class CorstonePerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], NPUCycles]
):
    """Corstone-based performance estimator."""

    def __init__(
        self, context: Context, device: EthosUConfiguration, backend: str
    ) -> None:
        """Init Corstone-based performance estimator."""
        self.context = context
        self.device = device
        self.backend = backend

    def estimate(self, model: Path | ModelConfiguration) -> NPUCycles:
        """Estimate performance."""
        with log_action(f"Getting the performance metrics for '{self.backend}' ..."):
            logger.info(
                "WARNING: This task may require several minutes "
                "(press ctrl-c to interrupt)"
            )

            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            optimized_model_path = self.context.get_model_path(
                f"{model_path.stem}_vela.tflite"
            )

            vela.optimize_model(
                model_path, self.device.compiler_options, optimized_model_path
            )

            model_info = backend_manager.ModelInfo(model_path=optimized_model_path)
            device_info = backend_manager.DeviceInfo(
                device_type=self.device.target,  # type: ignore
                mac=self.device.mac,
            )

            corstone_perf_metrics = backend_manager.estimate_performance(
                model_info, device_info, self.backend
            )

            return NPUCycles(
                corstone_perf_metrics.npu_active_cycles,
                corstone_perf_metrics.npu_idle_cycles,
                corstone_perf_metrics.npu_total_cycles,
                corstone_perf_metrics.npu_axi0_rd_data_beat_received,
                corstone_perf_metrics.npu_axi0_wr_data_beat_written,
                corstone_perf_metrics.npu_axi1_rd_data_beat_received,
            )


class EthosUPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], PerformanceMetrics]
):
    """Ethos-U performance estimator."""

    def __init__(
        self,
        context: Context,
        device: EthosUConfiguration,
        backends: list[str] | None = None,
    ) -> None:
        """Init performance estimator."""
        self.context = context
        self.device = device
        if backends is None:
            backends = ["Vela"]  # Only Vela is always available as default
        for backend in backends:
            if backend != "Vela" and not backend_manager.is_supported(backend):
                raise ValueError(
                    f"Unsupported backend '{backend}'. "
                    f"Only 'Vela' and {backend_manager.supported_backends()} "
                    "are supported."
                )
        self.backends = set(backends)

    def estimate(self, model: Path | ModelConfiguration) -> PerformanceMetrics:
        """Estimate performance."""
        model_path = (
            Path(model.model_path) if isinstance(model, ModelConfiguration) else model
        )

        tflite_model = get_tflite_model(model_path, self.context)

        memory_usage = None
        npu_cycles = None

        for backend in self.backends:
            if backend == "Vela":
                vela_estimator = VelaPerformanceEstimator(self.context, self.device)
                memory_usage = vela_estimator.estimate(tflite_model)
            elif backend in backend_manager.supported_backends():
                corstone_estimator = CorstonePerformanceEstimator(
                    self.context, self.device, backend
                )
                npu_cycles = corstone_estimator.estimate(tflite_model)
            else:
                logger.warning(
                    "Backend '%s' is not supported for Ethos-U performance "
                    "estimation.",
                    backend,
                )

        return PerformanceMetrics(self.device, npu_cycles, memory_usage)
