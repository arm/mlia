# Copyright 2021, Arm Ltd.
"""Tests for metrics module."""
from typing import Union

import pandas as pd
import pytest
from mlia.config import EthosU55
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics


@pytest.mark.parametrize(
    "metric, dataframe",
    [
        (
            NPUCycles(1, 2, 3, 4, 5, 6),
            pd.DataFrame.from_records(
                [[1, 2, 3, 4, 5, 6]],
                columns=[
                    "NPU active cycles",
                    "NPU idle cycles",
                    "NPU total cycles",
                    "NPU AXI0 RD data beat received",
                    "NPU AXI0 WR data beat written",
                    "NPU AXI1 RD data beat received",
                ],
            ),
        ),
        (
            MemoryUsage(1, 2, 3, 4, 5),
            pd.DataFrame.from_records(
                [[1, 2, 3, 4, 5]],
                columns=[
                    "SRAM used (bytes)",
                    "DRAM used (bytes)",
                    "Unknown memory used (bytes)",
                    "On chip flash used (bytes)",
                    "Off chip flash used (bytes)",
                ],
            ),
        ),
        (
            MemoryUsage(1024, 1024, 1024, 1024, 1024).in_kilobytes(),
            pd.DataFrame.from_records(
                [[1.0, 1.0, 1.0, 1.0, 1.0]],
                columns=[
                    "SRAM used (KiB)",
                    "DRAM used (KiB)",
                    "Unknown memory used (KiB)",
                    "On chip flash used (KiB)",
                    "Off chip flash used (KiB)",
                ],
            ),
        ),
        (
            PerformanceMetrics(
                EthosU55(),
                NPUCycles(1, 2, 3, 4, 5, 6),
                MemoryUsage(1, 2, 3, 4, 5),
            ),
            pd.DataFrame.from_records(
                [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]],
                columns=[
                    "SRAM used (bytes)",
                    "DRAM used (bytes)",
                    "Unknown memory used (bytes)",
                    "On chip flash used (bytes)",
                    "Off chip flash used (bytes)",
                    "NPU active cycles",
                    "NPU idle cycles",
                    "NPU total cycles",
                    "NPU AXI0 RD data beat received",
                    "NPU AXI0 WR data beat written",
                    "NPU AXI1 RD data beat received",
                ],
            ),
        ),
        (
            PerformanceMetrics(
                EthosU55(),
                NPUCycles(1, 2, 3, 4, 5, 6),
                MemoryUsage(1024, 2 * 1024, 3 * 1024, 4 * 1024, 5 * 1024),
            ).in_kilobytes(),
            pd.DataFrame.from_records(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 1, 2, 3, 4, 5, 6]],
                columns=[
                    "SRAM used (KiB)",
                    "DRAM used (KiB)",
                    "Unknown memory used (KiB)",
                    "On chip flash used (KiB)",
                    "Off chip flash used (KiB)",
                    "NPU active cycles",
                    "NPU idle cycles",
                    "NPU total cycles",
                    "NPU AXI0 RD data beat received",
                    "NPU AXI0 WR data beat written",
                    "NPU AXI1 RD data beat received",
                ],
            ),
        ),
    ],
)
def test_object_to_dataframe_conversion(
    metric: Union[MemoryUsage, NPUCycles, PerformanceMetrics], dataframe: pd.DataFrame
) -> None:
    """Test object to dataframe conversion."""
    assert metric.to_df().equals(dataframe)
