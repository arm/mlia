# Copyright 2021, Arm Ltd.
"""Metrics module."""


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
        unknown_memory_area_size: int,
        sram_memory_area_size: int,
        dram_memory_area_size: int,
        on_chip_flash_memory_area_size: int,
        off_chip_flash_memory_area_size: int,
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

        self.unknown_memory_area_size = unknown_memory_area_size
        self.sram_memory_area_size = sram_memory_area_size
        self.dram_memory_area_size = dram_memory_area_size
        self.on_chip_flash_memory_area_size = on_chip_flash_memory_area_size
        self.off_chip_flash_memory_area_size = off_chip_flash_memory_area_size
