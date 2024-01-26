# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reports module."""
from __future__ import annotations

from typing import Any
from typing import cast

import pytest

from mlia.backend.vela.compat import NpuSupported
from mlia.backend.vela.compat import Operator
from mlia.backend.vela.performance import LayerPerfInfo
from mlia.backend.vela.performance import LayerwisePerfInfo
from mlia.core.reporting import CompoundReport
from mlia.core.reporting import Report
from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import MemorySizeType
from mlia.target.ethos_u.performance import MemoryUsage
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.target.ethos_u.reporters import ethos_u_formatters
from mlia.target.ethos_u.reporters import report_operators
from mlia.target.ethos_u.reporters import report_perf_metrics
from mlia.target.ethos_u.reporters import report_target_details
from mlia.target.registry import profile
from mlia.utils.console import remove_ascii_codes


# pylint: disable=line-too-long
@pytest.mark.parametrize(
    "perf_metrics, expected_plain_text, expected_json_dict",
    [
        (
            [
                PerformanceMetrics(
                    target_config=EthosUConfiguration.load_profile("ethos-u55-256"),
                    npu_cycles=None,
                    memory_usage=MemoryUsage(
                        sram_memory_area_size=10,
                        dram_memory_area_size=0,
                        on_chip_flash_memory_area_size=0,
                        off_chip_flash_memory_area_size=20,
                        memory_size_type=MemorySizeType.KILOBYTES,
                    ),
                    layerwise_perf_info=LayerwisePerfInfo(
                        layerwise_info=[
                            LayerPerfInfo(
                                name="Test Layer",
                                tflite_operator="test_operator",
                                sram_usage=0,
                                op_cycles=0.0,
                                npu_cycles=0.0,
                                sram_access_cycles=0.0,
                                dram_access_cycles=0.0,
                                on_chip_flash_access_cycles=0.0,
                                off_chip_flash_access_cycles=0.0,
                                mac_count=0,
                                util_mac_percentage=0.0,
                            ),
                            LayerPerfInfo(
                                name="Test Layer 1",
                                tflite_operator="test_operator",
                                sram_usage=0,
                                op_cycles=0.0,
                                npu_cycles=0.0,
                                sram_access_cycles=0.0,
                                dram_access_cycles=0.0,
                                on_chip_flash_access_cycles=0.0,
                                off_chip_flash_access_cycles=0.0,
                                mac_count=0,
                                util_mac_percentage=0.0,
                            ),
                        ]
                    ),
                )
            ],
            """
Performance metrics:
┌─────────────────────┬──────────────┬──────┐
│ Metric              │ Value        │ Unit │
╞═════════════════════╪══════════════╪══════╡
│ SRAM used           │        10.00 │ KiB  │
├─────────────────────┼──────────────┼──────┤
│ Off-chip flash used │        20.00 │ KiB  │
└─────────────────────┴──────────────┴──────┘
IMPORTANT: The performance figures above refer to NPU only
Layer-Wise Metrics:
┌──────────────┬─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Layer Name   │ TFLite Operator │ SRAM Usage   │ OP Cycles    │ NPU Cycles   │ SRAM AC      │ DRAM AC      │ OnFlash AC   │ OffFlash AC  │ MAC Count    │ MAC Util (%) │
╞══════════════╪═════════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╡
│ Test Layer   │ test_operator   │            0 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │            0 │         0.00 │
├──────────────┼─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Test Layer 1 │ test_operator   │            0 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │            0 │         0.00 │
└──────────────┴─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
""".strip(),
            {
                "performance_metrics": [
                    {"metric": "SRAM used", "value": 10, "unit": "KiB"},
                    {"metric": "Off-chip flash used", "value": 20, "unit": "KiB"},
                ],
                "layerwise_metrics": [
                    {
                        "name": "Test Layer",
                        "tflite_operator": "test_operator",
                        "sram_usage": 0,
                        "op_cycles": 0.0,
                        "npu_cycles": 0.0,
                        "sram_access_cycles": 0.0,
                        "dram_access_cycles": 0.0,
                        "on_chip_flash_access_cycles": 0.0,
                        "off_chip_flash_access_cycles": 0.0,
                        "mac_count": 0,
                        "util_mac_percentage": 0.0,
                    },
                    {
                        "name": "Test Layer 1",
                        "tflite_operator": "test_operator",
                        "sram_usage": 0,
                        "op_cycles": 0.0,
                        "npu_cycles": 0.0,
                        "sram_access_cycles": 0.0,
                        "dram_access_cycles": 0.0,
                        "on_chip_flash_access_cycles": 0.0,
                        "off_chip_flash_access_cycles": 0.0,
                        "mac_count": 0,
                        "util_mac_percentage": 0.0,
                    },
                ],
            },
        ),
        (
            [
                PerformanceMetrics(
                    target_config=EthosUConfiguration.load_profile("ethos-u55-256"),
                    npu_cycles=None,
                    memory_usage=MemoryUsage(
                        sram_memory_area_size=10,
                        dram_memory_area_size=0,
                        on_chip_flash_memory_area_size=0,
                        off_chip_flash_memory_area_size=20,
                        memory_size_type=MemorySizeType.KILOBYTES,
                    ),
                    layerwise_perf_info=LayerwisePerfInfo(
                        layerwise_info=[
                            LayerPerfInfo(
                                name="Test Layer",
                                tflite_operator="test_operator",
                                sram_usage=0,
                                op_cycles=0.0,
                                npu_cycles=0.0,
                                sram_access_cycles=0.0,
                                dram_access_cycles=0.0,
                                on_chip_flash_access_cycles=0.0,
                                off_chip_flash_access_cycles=0.0,
                                mac_count=0,
                                util_mac_percentage=0.0,
                            ),
                            LayerPerfInfo(
                                name="Test Layer",
                                tflite_operator="test_operator",
                                sram_usage=0,
                                op_cycles=0.0,
                                npu_cycles=0.0,
                                sram_access_cycles=0.0,
                                dram_access_cycles=0.0,
                                on_chip_flash_access_cycles=0.0,
                                off_chip_flash_access_cycles=0.0,
                                mac_count=0,
                                util_mac_percentage=0.0,
                            ),
                        ]
                    ),
                )
            ],
            """
Performance metrics:
┌─────────────────────┬──────────────┬──────┐
│ Metric              │ Value        │ Unit │
╞═════════════════════╪══════════════╪══════╡
│ SRAM used           │        10.00 │ KiB  │
├─────────────────────┼──────────────┼──────┤
│ Off-chip flash used │        20.00 │ KiB  │
└─────────────────────┴──────────────┴──────┘
IMPORTANT: The performance figures above refer to NPU only
Layer-Wise Metrics:
┌────────────────┬─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Layer Name     │ TFLite Operator │ SRAM Usage   │ OP Cycles    │ NPU Cycles   │ SRAM AC      │ DRAM AC      │ OnFlash AC   │ OffFlash AC  │ MAC Count    │ MAC Util (%) │
╞════════════════╪═════════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╪══════════════╡
│ Test Layer     │ test_operator   │            0 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │            0 │         0.00 │
├────────────────┼─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Test Layer (1) │ test_operator   │            0 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │         0.00 │            0 │         0.00 │
└────────────────┴─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
""".strip(),
            {
                "performance_metrics": [
                    {"metric": "SRAM used", "value": 10, "unit": "KiB"},
                    {"metric": "Off-chip flash used", "value": 20, "unit": "KiB"},
                ],
                "layerwise_metrics": [
                    {
                        "name": "Test Layer",
                        "tflite_operator": "test_operator",
                        "sram_usage": 0,
                        "op_cycles": 0.0,
                        "npu_cycles": 0.0,
                        "sram_access_cycles": 0.0,
                        "dram_access_cycles": 0.0,
                        "on_chip_flash_access_cycles": 0.0,
                        "off_chip_flash_access_cycles": 0.0,
                        "mac_count": 0,
                        "util_mac_percentage": 0.0,
                    },
                    {
                        "name": "Test Layer (1)",
                        "tflite_operator": "test_operator",
                        "sram_usage": 0,
                        "op_cycles": 0.0,
                        "npu_cycles": 0.0,
                        "sram_access_cycles": 0.0,
                        "dram_access_cycles": 0.0,
                        "on_chip_flash_access_cycles": 0.0,
                        "off_chip_flash_access_cycles": 0.0,
                        "mac_count": 0,
                        "util_mac_percentage": 0.0,
                    },
                ],
            },
        ),
    ],
)
# pylint: enable=line-too-long
def test_report_perf_metrics(
    perf_metrics: PerformanceMetrics,
    expected_plain_text: str,
    expected_json_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test report_perf_metrics formatter."""
    monkeypatch.setenv("COLUMNS", "5000")
    report = report_perf_metrics(perf_metrics)
    assert isinstance(report, CompoundReport)
    plain_text = remove_ascii_codes(report.to_plain_text())
    assert plain_text == expected_plain_text
    json_dict = report.to_json()
    assert json_dict == expected_json_dict


@pytest.mark.parametrize(
    "ops, expected_plain_text, expected_json_dict",
    [
        (
            [
                Operator(
                    "npu_supported",
                    "test_type",
                    NpuSupported(True, []),
                ),
                Operator(
                    "cpu_only",
                    "test_type",
                    NpuSupported(
                        False,
                        [
                            (
                                "CPU only operator",
                                "",
                            ),
                        ],
                    ),
                ),
                Operator(
                    "npu_unsupported",
                    "test_type",
                    NpuSupported(
                        False,
                        [
                            (
                                "Not supported operator",
                                "Reason why operator is not supported",
                            )
                        ],
                    ),
                ),
            ],
            """
Operators:
┌───┬─────────────────┬───────────────┬───────────┬───────────────────────────────┐
│ # │ Operator name   │ Operator type │ Placement │ Notes                         │
╞═══╪═════════════════╪═══════════════╪═══════════╪═══════════════════════════════╡
│ 1 │ npu_supported   │ test_type     │ NPU       │                               │
├───┼─────────────────┼───────────────┼───────────┼───────────────────────────────┤
│ 2 │ cpu_only        │ test_type     │ CPU       │ * CPU only operator           │
├───┼─────────────────┼───────────────┼───────────┼───────────────────────────────┤
│ 3 │ npu_unsupported │ test_type     │ CPU       │ * Not supported operator      │
│   │                 │               │           │ * Reason why operator is not  │
│   │                 │               │           │ supported                     │
└───┴─────────────────┴───────────────┴───────────┴───────────────────────────────┘
""".strip(),
            {
                "operators": [
                    {
                        "operator_name": "npu_supported",
                        "operator_type": "test_type",
                        "placement": "NPU",
                        "notes": [],
                    },
                    {
                        "operator_name": "cpu_only",
                        "operator_type": "test_type",
                        "placement": "CPU",
                        "notes": [{"note": "CPU only operator"}],
                    },
                    {
                        "operator_name": "npu_unsupported",
                        "operator_type": "test_type",
                        "placement": "CPU",
                        "notes": [
                            {"note": "Not supported operator"},
                            {"note": "Reason why operator is not supported"},
                        ],
                    },
                ]
            },
        ),
    ],
)
def test_report_operators(
    ops: list[Operator],
    expected_plain_text: str,
    expected_json_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test report_operatos formatter."""
    # make terminal wide enough to print whole table
    monkeypatch.setenv("COLUMNS", "100")

    report = report_operators(ops)
    assert isinstance(report, Table)

    plain_text = remove_ascii_codes(report.to_plain_text())
    assert plain_text == expected_plain_text

    json_dict = report.to_json()
    assert json_dict == expected_json_dict


@pytest.mark.parametrize(
    "target, expected_plain_text, expected_json_dict",
    [
        [
            cast(EthosUConfiguration, profile("ethos-u55-256")),
            """Target information:
  Target                                                     ethos-u55
  MAC                                                              256

  Memory mode                                              Shared_Sram
    Const mem area                                                Axi1
    Arena mem area                                                Axi0
    Cache mem area                                                Axi0

  System config                            Ethos_U55_High_End_Embedded
    Accelerator clock                                   500,000,000 Hz
    AXI0 port                                                     Sram
    AXI1 port                                             OffChipFlash

    Memory area settings:
      Sram:
        Clock scales                                               1.0
        Burst length                                          32 bytes
        Read latency                                         32 cycles
        Write latency                                        32 cycles

      OffChipFlash:
        Clock scales                                             0.125
        Burst length                                         128 bytes
        Read latency                                         64 cycles
        Write latency                                        64 cycles""",
            {
                "target": {
                    "target": "ethos-u55",
                    "mac": 256,
                    "memory_mode": {
                        "const_mem_area": "Axi1",
                        "arena_mem_area": "Axi0",
                        "cache_mem_area": "Axi0",
                    },
                    "system_config": {
                        "accelerator_clock": {"value": 500000000.0, "unit": "Hz"},
                        "axi0_port": "Sram",
                        "axi1_port": "OffChipFlash",
                        "memory_area": {
                            "Sram": {
                                "clock_scales": 1.0,
                                "burst_length": {"value": 32, "unit": "bytes"},
                                "read_latency": {"value": 32, "unit": "cycles"},
                                "write_latency": {"value": 32, "unit": "cycles"},
                            },
                            "OffChipFlash": {
                                "clock_scales": 0.125,
                                "burst_length": {"value": 128, "unit": "bytes"},
                                "read_latency": {"value": 64, "unit": "cycles"},
                                "write_latency": {"value": 64, "unit": "cycles"},
                            },
                        },
                    },
                }
            },
        ],
    ],
)
def test_report_target_details(
    target: EthosUConfiguration,
    expected_plain_text: str,
    expected_json_dict: dict,
) -> None:
    """Test report_operatos formatter."""
    report = report_target_details(target)
    assert isinstance(report, Report)

    plain_text = report.to_plain_text()
    assert plain_text == expected_plain_text

    json_dict = report.to_json()
    assert json_dict == expected_json_dict


@pytest.mark.parametrize(
    "data",
    (TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE),),
)
def test_ethos_u_formatters(data: Any) -> None:
    """Test function ethos_u_formatters() with valid input."""
    formatter = ethos_u_formatters(data)
    report = formatter(data)
    assert isinstance(report, Report)


def test_ethos_u_formatters_invalid_data() -> None:
    """Test function ethos_u_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        ethos_u_formatters(200)
