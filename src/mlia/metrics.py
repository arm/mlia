# Copyright 2021, Arm Ltd.
"""Metrics module."""
from typing import Any
from typing import Iterator

import pandas as pd
from mlia.config import EthosUConfiguration


class NPUCycles:
    """NPU cycles metrics."""

    def __init__(
        self,
        npu_active_cycles: int,
        npu_idle_cycles: int,
        npu_total_cycles: int,
        npu_axi0_rd_data_beat_received: int,
        npu_axi0_wr_data_beat_written: int,
        npu_axi1_rd_data_beat_received: int,
    ):
        """Init NPU cycles metrics instance."""
        self.npu_active_cycles = npu_active_cycles
        self.npu_idle_cycles = npu_idle_cycles
        self.npu_total_cycles = npu_total_cycles
        self.npu_axi0_rd_data_beat_received = npu_axi0_rd_data_beat_received
        self.npu_axi0_wr_data_beat_written = npu_axi0_wr_data_beat_written
        self.npu_axi1_rd_data_beat_received = npu_axi1_rd_data_beat_received

    def __iter__(self) -> Iterator[Any]:
        """Class iterator to convert to/from dictionary."""
        return ((field, [value]) for field, value in vars(self).items())


class MemoryUsage:
    """Memory usage metrics."""

    def __init__(
        self,
        sram_memory_area_size: int,
        dram_memory_area_size: int,
        unknown_memory_area_size: int,
        on_chip_flash_memory_area_size: int,
        off_chip_flash_memory_area_size: int,
    ):
        """Init memory usage metrics instance."""
        self.sram_memory_area_size = sram_memory_area_size
        self.dram_memory_area_size = dram_memory_area_size
        self.unknown_memory_area_size = unknown_memory_area_size
        self.on_chip_flash_memory_area_size = on_chip_flash_memory_area_size
        self.off_chip_flash_memory_area_size = off_chip_flash_memory_area_size

    def __iter__(self) -> Iterator[Any]:
        """Class iterator to convert to/from dictionary."""
        return ((field, [value]) for field, value in vars(self).items())


class PerformanceMetrics:
    """Performance metrics."""

    row_names = [
        "SRAM used (KiB)",
        "DRAM used (KiB)",
        "Unknown memory used (KiB)",
        "On chip flash used (KiB)",
        "Off chip flash used (KiB)",
        "NPU active cycles (cycles)",
        "NPU idle cycles (cycles)",
        "NPU total cycles (cycles)",
        "NPU AXI0 RD data beat received (beats)",
        "NPU AXI0 WR data beat written (beats)",
        "NPU AXI1 RD data beat received (beats)",
    ]

    def __init__(
        self,
        device: EthosUConfiguration,
        npu_cycles: NPUCycles,
        memory_usage: MemoryUsage,
    ):
        """Initialize the performance metrics instance."""
        self.device = device
        self.npu_cycles = npu_cycles
        self.memory_usage = memory_usage

    def __iter__(self) -> Iterator[Any]:
        """Class iterator to convert to/from dictionary."""
        return iter([*self.memory_usage, *self.npu_cycles])

    def to_df(self) -> pd.DataFrame:
        """Convert object instance to Pandas dataframe."""
        return pd.DataFrame.from_dict(dict(self))
