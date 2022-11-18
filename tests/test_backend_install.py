# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for common management functionality."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.install import BackendInfo
from mlia.backend.install import get_all_application_names
from mlia.backend.install import get_all_system_names
from mlia.backend.install import get_system_name
from mlia.backend.install import is_supported
from mlia.backend.install import StaticPathChecker
from mlia.backend.install import supported_backends


@pytest.mark.parametrize(
    "copy_source, system_config",
    [
        (True, "system_config.json"),
        (True, None),
        (False, "system_config.json"),
        (False, None),
    ],
)
def test_static_path_checker(
    tmp_path: Path, copy_source: bool, system_config: str
) -> None:
    """Test static path checker."""
    checker = StaticPathChecker(tmp_path, ["file1.txt"], copy_source, system_config)
    tmp_path.joinpath("file1.txt").touch()

    result = checker(tmp_path)
    assert result == BackendInfo(tmp_path, copy_source, system_config)


def test_static_path_checker_invalid_path(tmp_path: Path) -> None:
    """Test static path checker with invalid path."""
    checker = StaticPathChecker(tmp_path, ["file1.txt"])

    result = checker(tmp_path)
    assert result is None

    result = checker(tmp_path / "unknown_directory")
    assert result is None


def test_supported_backends() -> None:
    """Test function supported backends."""
    assert supported_backends() == ["Corstone-300", "Corstone-310"]


@pytest.mark.parametrize(
    "backend, expected_result",
    [
        ["unknown_backend", False],
        ["Corstone-300", True],
        ["Corstone-310", True],
    ],
)
def test_is_supported(backend: str, expected_result: bool) -> None:
    """Test function is_supported."""
    assert is_supported(backend) == expected_result


@pytest.mark.parametrize(
    "backend, expected_result",
    [
        [
            "Corstone-300",
            [
                "Corstone-300: Cortex-M55+Ethos-U55",
                "Corstone-300: Cortex-M55+Ethos-U65",
            ],
        ],
        [
            "Corstone-310",
            [
                "Corstone-310: Cortex-M85+Ethos-U55",
                "Corstone-310: Cortex-M85+Ethos-U65",
            ],
        ],
    ],
)
def test_get_all_system_names(backend: str, expected_result: list[str]) -> None:
    """Test function get_all_system_names."""
    assert sorted(get_all_system_names(backend)) == expected_result


@pytest.mark.parametrize(
    "backend, expected_result",
    [
        [
            "Corstone-300",
            [
                "Generic Inference Runner: Ethos-U55",
                "Generic Inference Runner: Ethos-U65",
            ],
        ],
        [
            "Corstone-310",
            [
                "Generic Inference Runner: Ethos-U55",
                "Generic Inference Runner: Ethos-U65",
            ],
        ],
    ],
)
def test_get_all_application_names(backend: str, expected_result: list[str]) -> None:
    """Test function get_all_application_names."""
    assert sorted(get_all_application_names(backend)) == expected_result


def test_get_system_name() -> None:
    """Test function get_system_name."""
    assert (
        get_system_name("Corstone-300", "ethos-u55")
        == "Corstone-300: Cortex-M55+Ethos-U55"
    )

    with pytest.raises(KeyError):
        get_system_name("some_backend", "some_type")
