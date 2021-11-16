# Copyright 2021, Arm Ltd.
"""Tests for reports module."""
# pylint: disable=too-many-arguments
import sys
from contextlib import ExitStack as doesnt_raise
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal

import pytest
from mlia.config import EthosU55
from mlia.metadata import NpuSupported
from mlia.metadata import Operator
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics
from mlia.reporters import produce_report
from mlia.reporters import report_dataframe
from mlia.reporters import report_operators
from mlia.reporters import report_perf_metrics
from mlia.reporting import Table


@pytest.mark.parametrize(
    "data, formatters",
    [
        (
            [Operator("test_operator", "test_type", NpuSupported(False, []))],
            [report_operators, None],
        ),
        (
            PerformanceMetrics(
                EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
            ),
            [report_perf_metrics, None],
        ),
        (
            PerformanceMetrics(
                EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
            ).to_df(),
            [report_dataframe, None],
        ),
        (
            [
                (
                    [Operator("test_operator", "test_type", NpuSupported(False, []))],
                    PerformanceMetrics(
                        EthosU55(),
                        NPUCycles(0, 0, 0, 0, 0, 0),
                        MemoryUsage(0, 0, 0, 0, 0),
                    ),
                )
            ],
            [None],
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
    formatters: List[Callable],
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
                Operator("npu_supported", "test_type", NpuSupported(True, [])),
                Operator(
                    "cpu_only",
                    "test_type",
                    NpuSupported(False, [("CPU only operator", "")]),
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
╒═════╤═════════════════╤═════════════════╤═════════════╤══════════════════════════════╕
│ #   │ Operator name   │ Operator type   │ Placement   │ Notes                        │
╞═════╪═════════════════╪═════════════════╪═════════════╪══════════════════════════════╡
│ 1   │ npu_supported   │ test_type       │ NPU         │                              │
├─────┼─────────────────┼─────────────────┼─────────────┼──────────────────────────────┤
│ 2   │ cpu_only        │ test_type       │ CPU         │ * CPU only operator          │
├─────┼─────────────────┼─────────────────┼─────────────┼──────────────────────────────┤
│ 3   │ npu_unsupported │ test_type       │ CPU         │ * Not supported operator     │
│     │                 │                 │             │ * Reason why operator is not │
│     │                 │                 │             │ supported                    │
╘═════╧═════════════════╧═════════════════╧═════════════╧══════════════════════════════╛
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
    ops: List[Operator],
    expected_plain_text: str,
    expected_json_dict: Dict,
    expected_csv_list: List,
) -> None:
    """Test report_operatos formatter."""
    report = report_operators(ops)
    assert isinstance(report, Table)

    plain_text = report.to_plain_text()
    assert plain_text == expected_plain_text

    json_dict = report.to_json()
    assert json_dict == expected_json_dict

    csv_list = report.to_csv()
    assert csv_list == expected_csv_list
