# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reports module."""
from __future__ import annotations

import json
import sys
from contextlib import ExitStack as doesnt_raise
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal

import pytest

from mlia.core.reporting import get_reporter
from mlia.core.reporting import produce_report
from mlia.core.reporting import Report
from mlia.core.reporting import Reporter
from mlia.core.reporting import Table
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.performance import MemoryUsage
from mlia.devices.ethosu.performance import NPUCycles
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.devices.ethosu.reporters import ethos_u_formatters
from mlia.devices.ethosu.reporters import report_device_details
from mlia.devices.ethosu.reporters import report_operators
from mlia.devices.ethosu.reporters import report_perf_metrics
from mlia.tools.vela_wrapper import NpuSupported
from mlia.tools.vela_wrapper import Operator
from mlia.tools.vela_wrapper import Operators
from mlia.utils.console import remove_ascii_codes


@pytest.mark.parametrize(
    "data, formatters",
    [
        (
            [Operator("test_operator", "test_type", NpuSupported(False, []))],
            [report_operators],
        ),
        (
            PerformanceMetrics(
                EthosUConfiguration("ethos-u55-256"),
                NPUCycles(0, 0, 0, 0, 0, 0),
                MemoryUsage(0, 0, 0, 0, 0),
            ),
            [report_perf_metrics],
        ),
    ],
)
@pytest.mark.parametrize(
    "fmt, output, expected_error",
    [
        [
            "unknown_format",
            sys.stdout,
            pytest.raises(Exception, match="Unknown format unknown_format"),
        ],
        [
            "plain_text",
            sys.stdout,
            doesnt_raise(),
        ],
        [
            "json",
            sys.stdout,
            doesnt_raise(),
        ],
        [
            "csv",
            sys.stdout,
            doesnt_raise(),
        ],
        [
            "plain_text",
            "report.txt",
            doesnt_raise(),
        ],
        [
            "json",
            "report.json",
            doesnt_raise(),
        ],
        [
            "csv",
            "report.csv",
            doesnt_raise(),
        ],
    ],
)
def test_report(
    data: Any,
    formatters: list[Callable],
    fmt: Literal["plain_text", "json", "csv"],
    output: Any,
    expected_error: Any,
    tmp_path: Path,
) -> None:
    """Test report function."""
    if is_file := isinstance(output, str):
        output = tmp_path / output

    for formatter in formatters:
        with expected_error:
            produce_report(data, formatter, fmt, output)

            if is_file:
                assert output.is_file()
                assert output.stat().st_size > 0


@pytest.mark.parametrize(
    "ops, expected_plain_text, expected_json_dict, expected_csv_list",
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
│   │                 │               │           │                               │
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
            [
                ["Operator name", "Operator type", "Placement", "Notes"],
                ["npu_supported", "test_type", "NPU", ""],
                ["cpu_only", "test_type", "CPU", "CPU only operator"],
                [
                    "npu_unsupported",
                    "test_type",
                    "CPU",
                    "Not supported operator;Reason why operator is not supported",
                ],
            ],
        ),
    ],
)
def test_report_operators(
    ops: list[Operator],
    expected_plain_text: str,
    expected_json_dict: dict,
    expected_csv_list: list,
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

    csv_list = report.to_csv()
    assert csv_list == expected_csv_list


@pytest.mark.parametrize(
    "device, expected_plain_text, expected_json_dict, expected_csv_list",
    [
        [
            EthosUConfiguration("ethos-u55-256"),
            """Device information:
  Target                                                     ethos-u55
  MAC                                                              256

  Memory mode                                              Shared_Sram
    Const mem area                                                Axi1
    Arena mem area                                                Axi0
    Cache mem area                                                Axi0
    Arena cache size                                   2,096,768 bytes

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

      Dram:
        Clock scales                                               1.0
        Burst length                                            1 byte
        Read latency                                          0 cycles
        Write latency                                         0 cycles

      OnChipFlash:
        Clock scales                                               1.0
        Burst length                                            1 byte
        Read latency                                          0 cycles
        Write latency                                         0 cycles

      OffChipFlash:
        Clock scales                                             0.125
        Burst length                                         128 bytes
        Read latency                                         64 cycles
        Write latency                                        64 cycles

  Architecture settings:
    Permanent storage mem area                            OffChipFlash
    Feature map storage mem area                                  Sram
    Fast storage mem area                                         Sram""",
            {
                "device": {
                    "target": "ethos-u55",
                    "mac": 256,
                    "memory_mode": {
                        "const_mem_area": "Axi1",
                        "arena_mem_area": "Axi0",
                        "cache_mem_area": "Axi0",
                        "arena_cache_size": {"value": 2096768, "unit": "bytes"},
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
                            "Dram": {
                                "clock_scales": 1.0,
                                "burst_length": {"value": 1, "unit": "byte"},
                                "read_latency": {"value": 0, "unit": "cycles"},
                                "write_latency": {"value": 0, "unit": "cycles"},
                            },
                            "OnChipFlash": {
                                "clock_scales": 1.0,
                                "burst_length": {"value": 1, "unit": "byte"},
                                "read_latency": {"value": 0, "unit": "cycles"},
                                "write_latency": {"value": 0, "unit": "cycles"},
                            },
                            "OffChipFlash": {
                                "clock_scales": 0.125,
                                "burst_length": {"value": 128, "unit": "bytes"},
                                "read_latency": {"value": 64, "unit": "cycles"},
                                "write_latency": {"value": 64, "unit": "cycles"},
                            },
                        },
                    },
                    "arch_settings": {
                        "permanent_storage_mem_area": "OffChipFlash",
                        "feature_map_storage_mem_area": "Sram",
                        "fast_storage_mem_area": "Sram",
                    },
                }
            },
            [
                (
                    "target",
                    "mac",
                    "memory_mode",
                    "const_mem_area",
                    "arena_mem_area",
                    "cache_mem_area",
                    "arena_cache_size_value",
                    "arena_cache_size_unit",
                    "system_config",
                    "accelerator_clock_value",
                    "accelerator_clock_unit",
                    "axi0_port",
                    "axi1_port",
                    "clock_scales",
                    "burst_length_value",
                    "burst_length_unit",
                    "read_latency_value",
                    "read_latency_unit",
                    "write_latency_value",
                    "write_latency_unit",
                    "permanent_storage_mem_area",
                    "feature_map_storage_mem_area",
                    "fast_storage_mem_area",
                ),
                (
                    "ethos-u55",
                    256,
                    "Shared_Sram",
                    "Axi1",
                    "Axi0",
                    "Axi0",
                    2096768,
                    "bytes",
                    "Ethos_U55_High_End_Embedded",
                    500000000.0,
                    "Hz",
                    "Sram",
                    "OffChipFlash",
                    0.125,
                    128,
                    "bytes",
                    64,
                    "cycles",
                    64,
                    "cycles",
                    "OffChipFlash",
                    "Sram",
                    "Sram",
                ),
            ],
        ],
    ],
)
def test_report_device_details(
    device: EthosUConfiguration,
    expected_plain_text: str,
    expected_json_dict: dict,
    expected_csv_list: list,
) -> None:
    """Test report_operatos formatter."""
    report = report_device_details(device)
    assert isinstance(report, Report)

    plain_text = report.to_plain_text()
    assert plain_text == expected_plain_text

    json_dict = report.to_json()
    assert json_dict == expected_json_dict

    csv_list = report.to_csv()
    assert csv_list == expected_csv_list


def test_get_reporter(tmp_path: Path) -> None:
    """Test reporter functionality."""
    ops = Operators(
        [
            Operator(
                "npu_supported",
                "op_type",
                NpuSupported(True, []),
            ),
        ]
    )

    output = tmp_path / "output.json"
    with get_reporter("json", output, ethos_u_formatters) as reporter:
        assert isinstance(reporter, Reporter)

        with pytest.raises(
            Exception, match="Unable to find appropriate formatter for some_data"
        ):
            reporter.submit("some_data")

        reporter.submit(ops)

    with open(output, encoding="utf-8") as file:
        json_data = json.load(file)

        assert json_data == {
            "operators_stats": [
                {
                    "npu_unsupported_ratio": 0.0,
                    "num_of_npu_supported_operators": 1,
                    "num_of_operators": 1,
                }
            ]
        }
