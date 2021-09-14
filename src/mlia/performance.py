# Copyright 2021, Arm Ltd.
"""Performance estimation."""
import logging
from pathlib import Path
from typing import Optional

import mlia.tools.aiet_wrapper as aiet
import mlia.tools.vela_wrapper as vela
from mlia.config import EthosUConfiguration
from mlia.config import IPConfiguration
from mlia.config import ModelConfiguration
from mlia.config import TFLiteModel
from mlia.exceptions import ConfigurationError
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics


LOGGER = logging.getLogger("mlia.performance")


def collect_performance_metrics(
    model: ModelConfiguration,
    device: IPConfiguration,
    working_dir: Optional[str] = None,
) -> PerformanceMetrics:
    """Collect performance metrics."""
    if not isinstance(model, TFLiteModel):
        raise ConfigurationError("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise ConfigurationError("Unsupported device configuration")

    return ethosu_performance_metrics(model, device, working_dir)


def ethosu_performance_metrics(
    model: TFLiteModel, device: EthosUConfiguration, working_dir: Optional[str] = None
) -> PerformanceMetrics:
    """Return performance metrics for the EthousU device."""
    LOGGER.info("Getting the memory usage metrics ...")

    vela_perf_metrics = vela.estimate_performance(model, device)
    memory_usage = MemoryUsage(
        vela_perf_metrics.sram_memory_area_size,
        vela_perf_metrics.dram_memory_area_size,
        vela_perf_metrics.unknown_memory_area_size,
        vela_perf_metrics.on_chip_flash_memory_area_size,
        vela_perf_metrics.off_chip_flash_memory_area_size,
    )
    LOGGER.info("Done")

    LOGGER.info("Compiling the model ...")
    model_filename = f"{Path(model.model_path).stem}_vela.tflite"

    optimized_model_path = str(
        Path(working_dir) / model_filename if working_dir else model_filename
    )
    vela.optimize_model(model, device, optimized_model_path)
    LOGGER.info("Done")

    LOGGER.info("Getting the performance metrics ...")
    LOGGER.info(
        "WARNING: This task may require several minutes (press ctrl-c to interrupt)"
    )

    perf_metrics = aiet.estimate_performance(TFLiteModel(optimized_model_path), device)
    LOGGER.info("Done")

    npu_cycles = NPUCycles(
        perf_metrics.npu_active_cycles,
        perf_metrics.npu_idle_cycles,
        perf_metrics.npu_total_cycles,
        perf_metrics.npu_axi0_rd_data_beat_received,
        perf_metrics.npu_axi0_wr_data_beat_written,
        perf_metrics.npu_axi1_rd_data_beat_received,
    )

    return PerformanceMetrics(device, npu_cycles, memory_usage)
