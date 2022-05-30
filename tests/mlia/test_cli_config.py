# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.config module."""
from typing import List
from unittest.mock import MagicMock

import pytest

from mlia.cli.config import get_default_backends
from mlia.cli.config import is_corstone_backend


@pytest.mark.parametrize(
    "available_backends, expected_default_backends",
    [
        [["Vela"], ["Vela"]],
        [["Corstone-300"], ["Corstone-300"]],
        [["Corstone-310"], ["Corstone-310"]],
        [["Corstone-300", "Corstone-310"], ["Corstone-310"]],
        [["Vela", "Corstone-300", "Corstone-310"], ["Vela", "Corstone-310"]],
        [
            ["Vela", "Corstone-300", "Corstone-310", "New backend"],
            ["Vela", "Corstone-310", "New backend"],
        ],
        [
            ["Vela", "Corstone-300", "New backend"],
            ["Vela", "Corstone-300", "New backend"],
        ],
    ],
)
def test_get_default_backends(
    monkeypatch: pytest.MonkeyPatch,
    available_backends: List[str],
    expected_default_backends: List[str],
) -> None:
    """Test function get_default backends."""
    monkeypatch.setattr(
        "mlia.cli.config.get_available_backends",
        MagicMock(return_value=available_backends),
    )

    assert get_default_backends() == expected_default_backends


def test_is_corstone_backend() -> None:
    """Test function is_corstone_backend."""
    assert is_corstone_backend("Corstone-300") is True
    assert is_corstone_backend("Corstone-310") is True
    assert is_corstone_backend("New backend") is False
