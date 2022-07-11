# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the types related utility functions."""
from typing import Any
from typing import Iterable
from typing import Optional

import pytest

from mlia.utils.types import is_list_of
from mlia.utils.types import is_number
from mlia.utils.types import only_one_selected
from mlia.utils.types import parse_int


@pytest.mark.parametrize(
    "value, expected_result",
    [
        ["", False],
        ["abc", False],
        ["123", True],
        ["123.1", True],
        ["-123", True],
        ["-123.1", True],
        ["0", True],
        ["1.e10", True],
    ],
)
def test_is_number(value: str, expected_result: bool) -> None:
    """Test function is_number."""
    assert is_number(value) == expected_result


@pytest.mark.parametrize(
    "data, cls, elem_num, expected_result",
    [
        [(1, 2), int, 2, True],
        [[1, 2], int, 2, True],
        [[1, 2], int, 3, False],
        [["1", "2", "3"], str, None, True],
        [["1", "2", "3"], int, None, False],
    ],
)
def test_is_list(
    data: Any, cls: type, elem_num: Optional[int], expected_result: bool
) -> None:
    """Test function is_list."""
    assert is_list_of(data, cls, elem_num) == expected_result


@pytest.mark.parametrize(
    "options, expected_result",
    [
        [[True], True],
        [[False], False],
        [[True, True, False, False], False],
    ],
)
def test_only_one_selected(options: Iterable[bool], expected_result: bool) -> None:
    """Test function only_one_selected."""
    assert only_one_selected(*options) == expected_result


@pytest.mark.parametrize(
    "value, default, expected_int",
    [
        ["1", None, 1],
        ["abc", 123, 123],
        [None, None, None],
        [None, 11, 11],
    ],
)
def test_parse_int(
    value: Any, default: Optional[int], expected_int: Optional[int]
) -> None:
    """Test function parse_int."""
    assert parse_int(value, default) == expected_int
