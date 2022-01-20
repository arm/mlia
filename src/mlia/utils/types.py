# Copyright 2021, Arm Ltd.
"""Types related utility functions."""
from typing import Any
from typing import Optional


def is_list_of(data: Any, cls: type, elem_num: Optional[int] = None) -> bool:
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
