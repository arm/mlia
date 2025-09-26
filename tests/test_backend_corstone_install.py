# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Corstone related installation functions.."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.backend.corstone.install import CorstoneFVP
from mlia.backend.corstone.install import CorstoneInstaller
from mlia.backend.corstone.install import get_corstone_installation
from mlia.backend.install import Installation


@pytest.mark.parametrize(
    "archive, sha256_hash, fvp_expected_files, expected_fvp_version,"
    " expected_vft_files",
    [
        [
            "Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz",
            "6ea4096ecf8a8c06d6e76e21cae494f0c7139374cb33f6bc3964d189b84539a9",
            [
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65",
            ],
            "11.24_13",
            [
                "VHT_Corstone_SSE-300_Ethos-U55",
                "VHT_Corstone_SSE-300_Ethos-U65",
            ],
        ],
        [
            "Corstone-310/FVP_Corstone_SSE-310_11.24_13_Linux64.tgz",
            "616ecc0e82067fe0684790cf99638b3496f9ead11051a58d766e8258e766c556",
            [
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310",
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310_Ethos-U65",
            ],
            "11.24_13",
            [
                "VHT_Corstone_SSE-310",
                "VHT_Corstone_SSE-310_Ethos-U65",
            ],
        ],
        [
            "Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64.tgz",
            "6986af8805de54fa8dcbc54ea2cd63b305ebf5f1c07d3cba09641e2f8cc4e2f5",
            [
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-320",
            ],
            "11.27_25",
            [
                "VHT_Corstone_SSE-320",
            ],
        ],
    ],
)
def test_corstone_fvp(
    archive: str,
    sha256_hash: str,
    fvp_expected_files: list[str],
    expected_fvp_version: str,
    expected_vft_files: list[str],
) -> None:
    """Test CorstoneFVP class"""
    corstone_fvp = CorstoneFVP(
        archive=archive, sha256_hash=sha256_hash, fvp_expected_files=fvp_expected_files
    )
    assert corstone_fvp.get_fvp_version() == expected_fvp_version
    for actual, expected in zip(
        corstone_fvp.get_vht_expected_files(), expected_vft_files
    ):
        assert actual == expected


def test_coverstone_fvp_no_version_found() -> None:
    """Test if CorstoneFVP raises RuntimeError if FVP version is not found"""
    corestone_fvp = CorstoneFVP(
        archive="Corstone-300/FVP_Corstone_SSE-300_xx.yy_zz_Linux64.tgz",
        sha256_hash="6ea4096ecf8a8c06d6e76e21cae494f0c7139374cb33f6bc3964d189b84539a9",
        fvp_expected_files=[
            "models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
            "models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65",
        ],
    )
    with pytest.raises(RuntimeError):
        corestone_fvp.get_fvp_version()


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

    # Incorrect installer name
    with pytest.raises(RuntimeError):
        CorstoneInstaller(name="bad_name")(eula_agreement, tmp_path)

    assert mock_check_call.mock_calls == expected_calls
