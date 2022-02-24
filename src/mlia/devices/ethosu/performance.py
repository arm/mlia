# Copyright (C) 2021-2022, Arm Ltd.
"""Performance estimation."""
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import mlia.tools.aiet_wrapper as aiet
import mlia.tools.vela_wrapper as vela
import pandas as pd
from mlia.core.context import Context
from mlia.core.errors import ConfigurationError
from mlia.core.performance import PerformanceEstimator
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.config import IPConfiguration
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings


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

    def to_df(self) -> pd.DataFrame:
        """Convert object instance to the Pandas dataframe."""
        return pd.DataFrame.from_records(
            [
                [
                    self.npu_active_cycles,
                    self.npu_idle_cycles,
                    self.npu_total_cycles,
                    self.npu_axi0_rd_data_beat_received,
                    self.npu_axi0_wr_data_beat_written,
                    self.npu_axi1_rd_data_beat_received,
                ]
            ],
            columns=[
                "NPU active cycles",
                "NPU idle cycles",
                "NPU total cycles",
                "NPU AXI0 RD data beat received",
                "NPU AXI0 WR data beat written",
                "NPU AXI1 RD data beat received",
            ],
        )


BYTES_PER_KILOBYTE = 1024


class MemorySizeType(Enum):
    """Memory size type enumeration."""

    BYTES = 0
    KILOBYTES = 1


@dataclass
class MemoryUsage:
    """Memory usage metrics."""

    sram_memory_area_size: Union[int, float]
    dram_memory_area_size: Union[int, float]
    unknown_memory_area_size: Union[int, float]
    on_chip_flash_memory_area_size: Union[int, float]
    off_chip_flash_memory_area_size: Union[int, float]
    memory_size_type: MemorySizeType = MemorySizeType.BYTES

    _default_columns = [
        "SRAM used",
        "DRAM used",
        "Unknown memory used",
        "On chip flash used",
        "Off chip flash used",
    ]

    def in_kilobytes(self) -> "MemoryUsage":
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

    def to_df(self) -> pd.DataFrame:
        """Convert object instance to the Pandas dataframe."""
        suffixes = {
            MemorySizeType.BYTES: "(bytes)",
            MemorySizeType.KILOBYTES: "(KiB)",
        }

        suffix = suffixes[self.memory_size_type]
        columns = [f"{c} {suffix}" for c in self._default_columns]

        return pd.DataFrame.from_records(
            [
                [
                    self.sram_memory_area_size,
                    self.dram_memory_area_size,
                    self.unknown_memory_area_size,
                    self.on_chip_flash_memory_area_size,
                    self.off_chip_flash_memory_area_size,
                ]
            ],
            columns=columns,
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    device: EthosUConfiguration
    npu_cycles: NPUCycles
    memory_usage: MemoryUsage

    def to_df(self) -> pd.DataFrame:
        """Convert object instance to Pandas dataframe."""
        return self.memory_usage.to_df().join(self.npu_cycles.to_df())

    def in_kilobytes(self) -> "PerformanceMetrics":
        """Return metrics with memory usage in KiB."""
        return PerformanceMetrics(
            self.device, self.npu_cycles, self.memory_usage.in_kilobytes()
        )


@dataclass
class OptimizationPerformanceMetrics:
    """Optimization performance metrics."""

    original_perf_metrics: PerformanceMetrics
    optimizations_perf_metrics: List[
        Tuple[List[OptimizationSettings], PerformanceMetrics]
    ]


def collect_performance_metrics(
    model: ModelConfiguration, device: IPConfiguration, context: Context
) -> PerformanceMetrics:
    """Collect performance metrics."""
    if not isinstance(model, TFLiteModel):
        raise ConfigurationError("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise ConfigurationError("Unsupported device configuration")

    if not model.model_path:
        raise Exception("Model path is not provided")

    estimator = EthosUPerformanceEstimator(context, device)
    return estimator.estimate(Path(model.model_path))


class VelaPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], vela.PerformanceMetrics]
):
    """Vela based performance estimator."""

    def __init__(self, context: Context, device: EthosUConfiguration) -> None:
        """Init Vela based performance estimator."""
        self.context = context
        self.device = device

    def estimate(
        self, model: Union[Path, ModelConfiguration]
    ) -> vela.PerformanceMetrics:
        """Estimate performance."""
        model_path = (
            Path(model.model_path) if isinstance(model, ModelConfiguration) else model
        )

        return vela.estimate_performance(model_path, self.device.compiler_options)


class AIETPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], aiet.PerformanceMetrics]
):
    """AIET based performance estimator."""

    def __init__(self, context: Context, device: EthosUConfiguration) -> None:
        """Init AIET based performance estimator."""
        self.context = context
        self.device = device

    def estimate(
        self, model: Union[Path, ModelConfiguration]
    ) -> aiet.PerformanceMetrics:
        """Estimate performance."""
        model_path = (
            Path(model.model_path) if isinstance(model, ModelConfiguration) else model
        )

        optimized_model_path = self.context.get_model_path(
            f"{model_path.stem}_vela.tflite"
        )

        vela.optimize_model(
            model_path, self.device.compiler_options, optimized_model_path
        )

        optimized_model = TFLiteModel(optimized_model_path)
        optimized_model_input = optimized_model.input_details()
        if not optimized_model_input:
            raise Exception(
                "Unable to get input details for the "
                f"model {optimized_model.model_path}"
            )

        model_info = aiet.ModelInfo(
            model_path=Path(optimized_model.model_path),
            input_shape=optimized_model_input[0]["shape"],
            input_dtype=optimized_model_input[0]["dtype"],
        )
        device_info = aiet.DeviceInfo(
            device_type=self.device.ip_class, mac=self.device.mac  # type: ignore
        )

        return aiet.estimate_performance(model_info, device_info)


class EthosUPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], PerformanceMetrics]
):
    """Ethos-U performance estimator."""

    def __init__(self, context: Context, device: EthosUConfiguration) -> None:
        """Init performance estimator."""
        self.context = context
        self.device = device

    def estimate(self, model: Union[Path, ModelConfiguration]) -> PerformanceMetrics:
        """Estimate performance."""
        model_path = (
            Path(model.model_path) if isinstance(model, ModelConfiguration) else model
        )

        tflite_model = get_tflite_model(model_path, self.context)

        vela_estimator = VelaPerformanceEstimator(self.context, self.device)

        logger.info("Getting the memory usage metrics ...")
        vela_perf_metrics = vela_estimator.estimate(tflite_model)
        logger.info("Done\n")

        aiet_estimator = AIETPerformanceEstimator(self.context, self.device)
        logger.info("Getting the performance metrics ...")
        logger.info(
            "WARNING: This task may require several minutes (press ctrl-c to interrupt)"
        )
        aiet_perf_metrics = aiet_estimator.estimate(tflite_model)
        logger.info("Done\n")

        memory_usage = MemoryUsage(
            vela_perf_metrics.sram_memory_area_size,
            vela_perf_metrics.dram_memory_area_size,
            vela_perf_metrics.unknown_memory_area_size,
            vela_perf_metrics.on_chip_flash_memory_area_size,
            vela_perf_metrics.off_chip_flash_memory_area_size,
        )

        npu_cycles = NPUCycles(
            aiet_perf_metrics.npu_active_cycles,
            aiet_perf_metrics.npu_idle_cycles,
            aiet_perf_metrics.npu_total_cycles,
            aiet_perf_metrics.npu_axi0_rd_data_beat_received,
            aiet_perf_metrics.npu_axi0_wr_data_beat_written,
            aiet_perf_metrics.npu_axi1_rd_data_beat_received,
        )

        return PerformanceMetrics(self.device, npu_cycles, memory_usage)
