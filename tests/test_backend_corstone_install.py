# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Corstone related installation functions.."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.backend.corstone.install import Corstone300Installer
from mlia.backend.corstone.install import get_corstone_installations
from mlia.backend.install import Installation


def test_get_corstone_installations() -> None:
    """Test function get_corstone_installations."""
    installations = get_corstone_installations()
    assert len(installations) == 2

    assert all(isinstance(item, Installation) for item in installations)


@pytest.mark.parametrize(
    "eula_agreement, expected_calls",
    [
        [True, [call(["./FVP_Corstone_SSE-300.sh", "-q", "-d", "corstone-300"])]],
        [
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
    ],
)
def test_corstone_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eula_agreement: bool,
    expected_calls: Any,
) -> None:
    """Test Corstone 300 installer."""
    mock_check_call = MagicMock()

    monkeypatch.setattr(
        "mlia.backend.corstone.install.subprocess.check_call", mock_check_call
    )

    installer = Corstone300Installer()
    installer(eula_agreement, tmp_path)

    assert mock_check_call.mock_calls == expected_calls
