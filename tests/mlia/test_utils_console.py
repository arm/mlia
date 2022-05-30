# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for console utility functions."""
from typing import Iterable
from typing import List
from typing import Optional

import pytest

from mlia.utils.console import apply_style
from mlia.utils.console import create_section_header
from mlia.utils.console import produce_table
from mlia.utils.console import remove_ascii_codes


@pytest.mark.parametrize(
    "rows, headers, table_style, expected_result",
    [
        [[], [], "no_borders", ""],
        [
            [["1", "2", "3"]],
            ["Col 1", "Col 2", "Col 3"],
            "default",
            """
┌───────┬───────┬───────┐
│ Col 1 │ Col 2 │ Col 3 │
╞═══════╪═══════╪═══════╡
│ 1     │ 2     │ 3     │
└───────┴───────┴───────┘
""".strip(),
        ],
        [
            [["1", "2", "3"]],
            ["Col 1", "Col 2", "Col 3"],
            "nested",
            "Col 1 Col 2 Col 3 \n                  \n1     2     3",
        ],
        [
            [["1", "2", "3"]],
            ["Col 1", "Col 2", "Col 3"],
            "no_borders",
            " Col 1  Col 2  Col 3 \n 1      2      3",
        ],
    ],
)
def test_produce_table(
    rows: Iterable, headers: Optional[List[str]], table_style: str, expected_result: str
) -> None:
    """Test produce_table function."""
    result = produce_table(rows, headers, table_style)
    assert remove_ascii_codes(result) == expected_result


def test_produce_table_unknown_style() -> None:
    """Test that function should fail if unknown style provided."""
    with pytest.raises(Exception, match="Unsupported table style unknown_style"):
        produce_table([["1", "2", "3"]], [], "unknown_style")


@pytest.mark.parametrize(
    "value, expected_result",
    [
        ["some text", "some text"],
        ["\033[32msome text\033[0m", "some text"],
    ],
)
def test_remove_ascii_codes(value: str, expected_result: str) -> None:
    """Test remove_ascii_codes function."""
    assert remove_ascii_codes(value) == expected_result


def test_apply_style() -> None:
    """Test function apply_style."""
    assert apply_style("some text", "green") == "[green]some text"


@pytest.mark.parametrize(
    "section_header, expected_result",
    [
        [
            "Section header",
            "\n--- Section header -------------------------------"
            "------------------------------\n",
        ],
        [
            "",
            f"\n{'-' * 80}\n",
        ],
    ],
)
def test_create_section_header(section_header: str, expected_result: str) -> None:
    """Test function test_create_section."""
    assert create_section_header(section_header) == expected_result


def test_create_section_header_too_long_value() -> None:
    """Test that header could not be created for the too long section names."""
    section_name = "section name" * 100
    with pytest.raises(ValueError, match="Section name too long"):
        create_section_header(section_name)
