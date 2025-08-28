# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Corstone related installation functions.."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.backend.corstone.install import CorstoneInstaller
from mlia.backend.corstone.install import get_corstone_installation
from mlia.backend.install import Installation


@pytest.mark.parametrize(
    "corstone_name", ["corstone-300", "corstone-310", "corstone-320"]
)
def test_get_corstone_installation(corstone_name: str) -> None:
    """Test Corstone installation"""
    installation = get_corstone_installation(corstone_name)
    assert isinstance(installation, Installation)


@pytest.mark.parametrize(
    "corstone_name, eula_agreement, expected_calls",
    [
        [
            "corstone-300",
            True,
            [call(["./FVP_Corstone_SSE-300.sh", "-q", "-d", "corstone-300"])],
        ],
        [
            "corstone-300",
            False,
            [
                call(
                    [
                        "./FVP_Corstone_SSE-300.sh",
                        "-q",
                        "-d",
                        "corstone-300",
                        "--nointeractive",
                        "--i-agree-to-the-contained-eula",
                    ]
                )
            ],
        ],
        [
            "corstone-310",
            True,
            [call(["./FVP_Corstone_SSE-310.sh", "-q", "-d", "corstone-310"])],
        ],
        [
            "corstone-310",
            False,
            [
                call(
                    [
                        "./FVP_Corstone_SSE-310.sh",
                        "-q",
                        "-d",
                        "corstone-310",
                        "--nointeractive",
                        "--i-agree-to-the-contained-eula",
                    ]
                )
            ],
        ],
        [
            "corstone-320",
            False,
            [
                call(
                    [
                        "./FVP_Corstone_SSE-320.sh",
                        "-q",
                        "-d",
                        "corstone-320",
                        "--nointeractive",
                        "--i-agree-to-the-contained-eula",
                    ]
                )
            ],
        ],
    ],
)
def test_corstone_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corstone_name: str,
    eula_agreement: bool,
    expected_calls: Any,
) -> None:
    """Test Corstone installer."""
    mock_check_call = MagicMock()

    monkeypatch.setattr(
        "mlia.backend.corstone.install.subprocess.check_call", mock_check_call
    )

    installer = CorstoneInstaller(name=corstone_name)
    installer(eula_agreement, tmp_path)

    assert mock_check_call.mock_calls == expected_calls
