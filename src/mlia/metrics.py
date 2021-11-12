# Copyright 2021, Arm Ltd.
"""Metrics module."""
from dataclasses import dataclass
from enum import Enum
from typing import Union

import pandas as pd
from mlia.config import EthosUConfiguration


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


class MemorySizeType(Enum):
    """Memory size type enumeration."""

    BYTES = 0
    KILOBYTES = 1


BYTES_PER_KILOBYTE = 1024


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
