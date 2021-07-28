# Copyright 2021, Arm Ltd.
"""Metrics module."""
from mlia.config import EthosUConfiguration


class NPUCycles:
    """NPU cycles metrics."""

    def __init__(
        self,
        npu_axi0_rd_data_beat_received: int,
        npu_axi0_wr_data_beat_written: int,
        npu_axi1_rd_data_beat_received: int,
        npu_active_cycles: int,
        npu_idle_cycles: int,
        npu_total_cycles: int,
    ):
        """Init NPU cycles metrics instance."""
        self.npu_axi0_rd_data_beat_received = npu_axi0_rd_data_beat_received
        self.npu_axi0_wr_data_beat_written = npu_axi0_wr_data_beat_written
        self.npu_axi1_rd_data_beat_received = npu_axi1_rd_data_beat_received
        self.npu_active_cycles = npu_active_cycles
        self.npu_idle_cycles = npu_idle_cycles
        self.npu_total_cycles = npu_total_cycles


class MemoryUsage:
    """Memory usage metrics."""

    def __init__(
        self,
        unknown_memory_area_size: int,
        sram_memory_area_size: int,
        dram_memory_area_size: int,
        on_chip_flash_memory_area_size: int,
        off_chip_flash_memory_area_size: int,
    ):
        """Init memory usage metrics instance."""
        self.unknown_memory_area_size = unknown_memory_area_size
        self.sram_memory_area_size = sram_memory_area_size
        self.dram_memory_area_size = dram_memory_area_size
        self.on_chip_flash_memory_area_size = on_chip_flash_memory_area_size
        self.off_chip_flash_memory_area_size = off_chip_flash_memory_area_size


class PerformanceMetrics:
    """Performance metrics."""

    def __init__(
        self,
        device: EthosUConfiguration,
        npu_cycles: NPUCycles,
        memory_usage: MemoryUsage,
    ) -> None:
        """Initialize the performance metrics instance."""
        self.device = device
        self.npu_cycles = npu_cycles
        self.memory_usage = memory_usage
