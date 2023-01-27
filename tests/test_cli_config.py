# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.config module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlia.cli.config import get_default_backends


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
        [["ArmNNTFLiteDelegate"], ["ArmNNTFLiteDelegate"]],
        [["tosa-checker"], ["tosa-checker"]],
        [
            ["ArmNNTFLiteDelegate", "Corstone-300"],
            ["ArmNNTFLiteDelegate", "Corstone-300"],
        ],
    ],
)
def test_get_default_backends(
    monkeypatch: pytest.MonkeyPatch,
    available_backends: list[str],
    expected_default_backends: list[str],
) -> None:
    """Test function get_default backends."""
    monkeypatch.setattr(
        "mlia.cli.config.get_available_backends",
        MagicMock(return_value=available_backends),
    )

    assert get_default_backends() == expected_default_backends
