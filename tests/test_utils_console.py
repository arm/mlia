# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for console utility functions."""

from __future__ import annotations

import sys
from typing import Iterable

import pytest

from mlia.utils.console import (
    apply_style,
    create_section_header,
    produce_table,
    remove_ascii_codes,
)


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
            "Col 1 Col 2 Col 3 \n1     2     3",
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
    rows: Iterable, headers: list[str] | None, table_style: str, expected_result: str
) -> None:
    """Test produce_table function."""
    result = produce_table(rows, headers, table_style)
    assert remove_ascii_codes(result) == expected_result


def test_produce_table_unknown_style() -> None:
    """Test that function should fail if unknown style provided."""
    with pytest.raises(ValueError, match="Table style unknown_style is not supported."):
        produce_table([["1", "2", "3"]], [], "unknown_style")


def test_produce_table_truncates_terminal_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redirected output should not truncate its output"""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    value = "x" * 100
    result = remove_ascii_codes(produce_table([[value]], table_style="no_borders"))

    assert value not in result
    assert "…" in result


def test_produce_table_does_not_truncate_redirected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redirected output should not truncate its output"""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    value = "x" * 100
    result = remove_ascii_codes(produce_table([[value]], table_style="no_borders"))

    assert value in result
    assert "…" not in result


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
            f"\n--- Section header {'-' * 101}\n",
        ],
        [
            "",
            f"\n{'-' * 120}\n",
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
