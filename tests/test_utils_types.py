# Copyright (C) 2021-2022, Arm Ltd.
"""Tests for the types related utility functions."""
from typing import Any
from typing import Optional

import pytest
from mlia.utils.types import is_list_of
from mlia.utils.types import is_number


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
