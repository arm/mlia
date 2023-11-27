# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra reporters."""
from functools import partial
from pathlib import Path
from typing import List

import pytest
from rich.console import Console

from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.performance import ArgoPerformanceMetrics
from mlia.backend.argo.performance import OperatorPerformanceData
from mlia.backend.ngp_graph_compiler.config import NGPGraphCompilerConfig
from mlia.backend.ngp_graph_compiler.output_parsing import NGPPerformanceDatabase
from mlia.backend.ngp_graph_compiler.performance import NGPGraphCompilerOutputFiles
from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceMetrics,
)
from mlia.core.reporting import Table
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.reporters import hydra_formatters
from mlia.target.hydra.reporters import report_target
from mlia.utils.console import remove_ascii_codes


def test_report_target() -> None:
    """Test function report_target()."""
    report = report_target(HydraConfiguration.load_profile("hydra"))
    assert report.to_plain_text()


def assert_table_contents(report: Table, json: dict) -> None:
    """Assert that a given Table renders the expected JSON output."""
    assert isinstance(report, Table)
    assert report.to_json() == json


def assert_table_lines(report: Table, expected_lines: list) -> None:
    """Assert that a given Table renders the expected JSON output.

    In case of failure, it renders actual and expected textual tables in a form
    that's easy to overview and can directly be used as "golden" data in the test.
    """
    assert isinstance(report, Table)
    actual_lines = remove_ascii_codes(report.to_plain_text()).split("\n")

    def to_diff_string(lines: List[str]) -> str:
        test_line = [f'          "{line}",' for line in lines]
        return ("\n").join(test_line)

    actual = to_diff_string(actual_lines)
    expected = to_diff_string(expected_lines)
    assert actual_lines == expected_lines, f"Expected:\n{expected}\n\nActual:\n{actual}"


def test_hydra_formatters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function hydra_formatters() with valid input."""
    op_performance_data = OperatorPerformanceData(
        "BiasAdd",
        "BIAS_ADD",
        [{"n_pass": 4, "hw_block": "SE", "duration": 5.123456789}],
    ).get_performance_metrics()
    assert op_performance_data

    metrics = ArgoPerformanceMetrics(
        backend_config=ArgoConfig(),
        metrics_file=Path("DOES_NOT_EXIST"),
        operator_performance_data=[
            OperatorPerformanceData(
                "Relu",
                "CONV_2D",
                [
                    {"n_pass": 1, "hw_block": "NE", "duration": 2.123456789},
                    {"n_pass": 2, "hw_block": "NE", "duration": 3.123456789},
                ],
            ),
            OperatorPerformanceData(
                "Relu",
                "DEPTHWISE_CONV_2D",
                [{"n_pass": 3, "hw_block": "SE", "duration": 4.123456789}],
            ),
            OperatorPerformanceData(
                "BiasAdd",
                "BIAS_ADD",
                [{"n_pass": 4, "hw_block": "SE", "duration": 5.123456789}],
            ),
        ],
    )

    monkeypatch.setattr("mlia.utils.console.Console", partial(Console, width=80))

    formatter = hydra_formatters(metrics)
    report = formatter(metrics)
    assert isinstance(report, Table)

    assert_table_lines(
        report,
        [
            # pylint: disable=C0301
            "Argo per-layer analysis:",
            "┌──────────────┬──────────────┬────────┬──────────┬──────────────┬─────────────┐",
            "│ Operator     │              │        │          │              │ Percentage  │",
            "│ name         │ Type         │ Pass # │ HW Block │ Duration(µs) │ of time     │",
            "╞══════════════╪══════════════╪════════╪══════════╪══════════════╪═════════════╡",
            "│ Relu         │ CONV_2D      │ 1      │ NE       │ 2.1235       │ 14.65%      │",
            "│              │              │ 2      │ NE       │ 3.1235       │ 21.55%      │",
            "├──────────────┼──────────────┼────────┼──────────┼──────────────┼─────────────┤",
            "│ BiasAdd      │ BIAS_ADD     │ 4      │ SE       │ 5.1235       │ 35.35%      │",
            "├──────────────┼──────────────┼────────┼──────────┼──────────────┼─────────────┤",
            "│ Relu         │ DEPTHWISE_C… │ 3      │ SE       │ 4.1235       │ 28.45%      │",
            "└──────────────┴──────────────┴────────┴──────────┴──────────────┴─────────────┘",
            # pylint: enable=C0301
        ],
    )


def test_ngp_graph_compiler_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function hydra_formatters() with NGP performance data."""

    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()

    performance_db = NGPPerformanceDatabase()
    performance_db.records = performance_db.parse_contents(contents)

    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NGPGraphCompilerConfig(sys_cfg, compiler_cfg)

    ignored_path = Path("ignored")

    metrics = NGPGraphCompilerPerformanceMetrics(
        backend_config=cfg,
        output_files=NGPGraphCompilerOutputFiles(
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
        ),
        performance_db=performance_db,
    )

    monkeypatch.setattr("mlia.utils.console.Console", partial(Console, width=80))

    formatter = hydra_formatters(metrics)
    report = formatter(metrics)
    assert isinstance(report, Table)

    assert_table_lines(
        report,
        [
            # pylint: disable=C0301
            "NGP raw performance report:",
            "┌────────┬────────┬────────┬────────┬────────┬────────┬───────┬────────┬───────┐",
            "│ Opera… │ Opera… │ Total  │ HW     │ HW     │ Memory │ Read  │ Write  │ Traf… │",
            "│ ID     │ Cycles │ Cycles │ Secti… │ Utili… │ Name   │ bytes │ bytes  │ cycl… │",
            "╞════════╪════════╪════════╪════════╪════════╪════════╪═══════╪════════╪═══════╡",
            "│ 25     │ 4      │ 13     │ Outpu… │ 0.0625 │ Undef… │ 0     │ 0      │ 0     │",
            "│        │        │        │ Vecto… │ 0.125  │ Inter… │ 0     │ 0      │ 0     │",
            "│        │        │        │ Vecto… │ 0.125  │ L1     │ 0     │ 4      │ 0     │",
            "│        │        │        │ Vecto… │ 0.125  │ L2     │ 0     │ 0      │ 0     │",
            "│        │        │        │ Vecto… │ 0.125  │ Syste… │ 0     │ 0      │ 0     │",
            "│        │        │        │ Input… │ 0.0625 │ DRAM   │ 128   │ 4      │ 4     │",
            "│        │        │        │ Input… │ 0.0625 │        │       │        │       │",
            "├────────┼────────┼────────┼────────┼────────┼────────┼───────┼────────┼───────┤",
            "│ 26     │ 18     │ 212    │ Outpu… │ 1      │ Undef… │ 0     │ 0      │ 0     │",
            "│        │        │        │ Vecto… │ 0.25   │ Inter… │ 0     │ 0      │ 0     │",
            "│        │        │        │ Vecto… │ 0.25   │ L1     │ 0     │ 0      │ 0     │",
            "│        │        │        │ Vecto… │ 0.25   │ L2     │ 0     │ 0      │ 0     │",
            "│        │        │        │ Trans… │ 0.25   │ Syste… │ 0     │ 0      │ 0     │",
            "│        │        │        │ Trans… │ 0.25   │ DRAM   │ 320   │ 12     │ 10    │",
            "│        │        │        │ Input… │ 0.0625 │        │       │        │       │",
            "│        │        │        │ Input… │ 0.0625 │        │       │        │       │",
            "│        │        │        │ Input… │ 0.25   │        │       │        │       │",
            "└────────┴────────┴────────┴────────┴────────┴────────┴───────┴────────┴───────┘",
            # pylint: enable=C0301
        ],
    )


def test_hydra_formatters_invalid_data() -> None:
    """Test hydra_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        hydra_formatters(12)


# %%
