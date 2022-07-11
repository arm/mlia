# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for misc util functions."""
from unittest.mock import MagicMock

import pytest

from mlia.utils.misc import yes


@pytest.mark.parametrize(
    "response, expected_result",
    [
        ["Y", True],
        ["y", True],
        ["N", False],
        ["n", False],
    ],
)
def test_yes(
    monkeypatch: pytest.MonkeyPatch, expected_result: bool, response: str
) -> None:
    """Test yes function."""
    monkeypatch.setattr("builtins.input", MagicMock(return_value=response))
    assert yes("some_prompt") == expected_result
