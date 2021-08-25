# Copyright 2021, Arm Ltd.
"""Metrics module."""
from enum import Enum
from typing import Union

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


class MemorySizeType(Enum):
    """Memory size type enumeration."""

    BYTES = 0
    KILOBYTES = 1


class MemoryUsage:
    """Memory usage metrics."""

    _default_columns = [
        "SRAM used",
        "DRAM used",
        "Unknown memory used",
        "On chip flash used",
        "Off chip flash used",
    ]

    def __init__(
        self,
        sram_memory_area_size: Union[int, float],
        dram_memory_area_size: Union[int, float],
        unknown_memory_area_size: Union[int, float],
        on_chip_flash_memory_area_size: Union[int, float],
        off_chip_flash_memory_area_size: Union[int, float],
        *,
        memory_size_type: MemorySizeType = MemorySizeType.BYTES,
    ):
        """Init memory usage metrics instance."""
        self.sram_memory_area_size = sram_memory_area_size
        self.dram_memory_area_size = dram_memory_area_size
        self.unknown_memory_area_size = unknown_memory_area_size
        self.on_chip_flash_memory_area_size = on_chip_flash_memory_area_size
        self.off_chip_flash_memory_area_size = off_chip_flash_memory_area_size
        self.memory_size_type = memory_size_type

    def in_kilobytes(self) -> "MemoryUsage":
        """Return memory usage with values in kilobytes."""
        if self.memory_size_type == MemorySizeType.KILOBYTES:
            return self

        kilobytes = [
            value / 1024
            for value in [
                self.sram_memory_area_size,
                self.dram_memory_area_size,
                self.unknown_memory_area_size,
                self.on_chip_flash_memory_area_size,
                self.off_chip_flash_memory_area_size,
            ]
        ]

        return MemoryUsage(*kilobytes, memory_size_type=MemorySizeType.KILOBYTES)

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


class PerformanceMetrics:
    """Performance metrics."""

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

    def to_df(self) -> pd.DataFrame:
        """Convert object instance to Pandas dataframe."""
        return self.memory_usage.to_df().join(self.npu_cycles.to_df())

    def in_kilobytes(self) -> "PerformanceMetrics":
        """Return metrics with memory usage in KiB."""
        return PerformanceMetrics(
            self.device, self.npu_cycles, self.memory_usage.in_kilobytes()
        )
