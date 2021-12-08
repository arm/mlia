# Copyright 2021, Arm Ltd.
"""Performance estimation."""
import logging
from pathlib import Path

import mlia.tools.aiet_wrapper as aiet
import mlia.tools.vela_wrapper as vela
from mlia.config import Context
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.config import IPConfiguration
from mlia.devices.ethosu.metrics import MemoryUsage
from mlia.devices.ethosu.metrics import NPUCycles
from mlia.devices.ethosu.metrics import PerformanceMetrics
from mlia.exceptions import ConfigurationError
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.config import TFLiteModel


logger = logging.getLogger(__name__)


def collect_performance_metrics(
    model: ModelConfiguration, device: IPConfiguration, ctx: Context
) -> PerformanceMetrics:
    """Collect performance metrics."""
    if not isinstance(model, TFLiteModel):
        raise ConfigurationError("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise ConfigurationError("Unsupported device configuration")

    return ethosu_performance_metrics(model, device, ctx)


def ethosu_performance_metrics(
    model: TFLiteModel, device: EthosUConfiguration, ctx: Context
) -> PerformanceMetrics:
    """Return performance metrics for the EthousU device."""
    logger.info("Getting the memory usage metrics ...")

    vela_perf_metrics = vela.estimate_performance(
        Path(model.model_path), device.compiler_options
    )
    memory_usage = MemoryUsage(
        vela_perf_metrics.sram_memory_area_size,
        vela_perf_metrics.dram_memory_area_size,
        vela_perf_metrics.unknown_memory_area_size,
        vela_perf_metrics.on_chip_flash_memory_area_size,
        vela_perf_metrics.off_chip_flash_memory_area_size,
    )
    logger.info("Done")

    logger.info("Compiling the model ...")
    model_filename = f"{Path(model.model_path).stem}_vela.tflite"

    optimized_model_path = ctx.get_model_path(model_filename)
    vela.optimize_model(
        Path(model.model_path), device.compiler_options, optimized_model_path
    )
    logger.info("Done")

    logger.info("Getting the performance metrics ...")
    logger.info(
        "WARNING: This task may require several minutes (press ctrl-c to interrupt)"
    )

    optimized_model = TFLiteModel(optimized_model_path)
    optimized_model_input = optimized_model.input_details()
    if not optimized_model_input:
        raise Exception(
            f"Unable to get input details for the model {optimized_model.model_path}"
        )

    model_info = aiet.ModelInfo(
        model_path=Path(optimized_model.model_path),
        input_shape=optimized_model_input[0]["shape"],
        input_dtype=optimized_model_input[0]["dtype"],
    )
    device_info = aiet.DeviceInfo(device_type=device.ip_class, mac=device.mac)
    perf_metrics = aiet.estimate_performance(model_info, device_info)
    logger.info("Done")

    npu_cycles = NPUCycles(
        perf_metrics.npu_active_cycles,
        perf_metrics.npu_idle_cycles,
        perf_metrics.npu_total_cycles,
        perf_metrics.npu_axi0_rd_data_beat_received,
        perf_metrics.npu_axi0_wr_data_beat_written,
        perf_metrics.npu_axi1_rd_data_beat_received,
    )

    return PerformanceMetrics(device, npu_cycles, memory_usage)
