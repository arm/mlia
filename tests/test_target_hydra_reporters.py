# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra reporters."""
from pathlib import Path
from typing import List

import pytest

from mlia.core.reporting import Table
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.performance import HydraPerformanceMetrics
from mlia.target.hydra.performance import OperatorPerformanceData
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


def test_hydra_formatters() -> None:
    """Test function hydra_formatters() with valid input."""
    metrics = HydraPerformanceMetrics(
        target_config=HydraConfiguration(target="hydra"),
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

    formatter = hydra_formatters(metrics)
    report = formatter(metrics)
    assert isinstance(report, Table)

    assert_table_lines(
        report,
        [
            "Argo per-layer analysis:",
            "┌───────────────┬───────────────────┬────────┬──────────┬──────────────┐",
            "│ Operator name │ Type              │ Pass # │ HW Block │ Duration(µs) │",
            "╞═══════════════╪═══════════════════╪════════╪══════════╪══════════════╡",
            "│ Relu          │ CONV_2D           │ 1      │ NE       │ 2.1235       │",
            "│               │                   │ 2      │ NE       │ 3.1235       │",
            "├───────────────┼───────────────────┼────────┼──────────┼──────────────┤",
            "│ Relu          │ DEPTHWISE_CONV_2D │ 3      │ SE       │ 4.1235       │",
            "├───────────────┼───────────────────┼────────┼──────────┼──────────────┤",
            "│ BiasAdd       │ BIAS_ADD          │ 4      │ SE       │ 5.1235       │",
            "└───────────────┴───────────────────┴────────┴──────────┴──────────────┘",
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
