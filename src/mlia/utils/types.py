# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Types related utility functions."""
from __future__ import annotations

from typing import Any


def is_list_of(data: Any, cls: type, elem_num: int | None = None) -> bool:
    """Check if data is a list of object of the same class."""
    return (
        isinstance(data, (tuple, list))
        and all(isinstance(item, cls) for item in data)
        and (elem_num is None or len(data) == elem_num)
    )


def is_number(value: str) -> bool:
    """Return true if string contains a number."""
    try:
        float(value)
    except ValueError:
        return False

    return True


def parse_int(value: Any, default: int | None = None) -> int | None:
    """Parse integer value."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def only_one_selected(*options: bool) -> bool:
    """Return true if only one True value found."""
    return sum(options) == 1
