# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for python package manager."""
import subprocess  # nosec
import sys
from unittest.mock import MagicMock

import pytest

from mlia.utils.py_manager import get_package_manager
from mlia.utils.py_manager import PyPackageManager


def test_get_package_manager() -> None:
    """Test function get_package_manager."""
    manager = get_package_manager()
    assert isinstance(manager, PyPackageManager)


@pytest.fixture(name="mock_check_output")
def mock_check_output_fixture(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock check_call function."""
    mock_check_output = MagicMock()

    monkeypatch.setattr(
        "mlia.utils.py_manager.subprocess.check_output", mock_check_output
    )

    return mock_check_output


def test_py_package_manager_metadata() -> None:
    """Test getting package status."""
    manager = PyPackageManager()
    assert manager.package_installed("pytest")
    assert manager.packages_installed(["pytest", "mlia"])


def test_py_package_manager_install(mock_check_output: MagicMock) -> None:
    """Test package installation."""
    manager = PyPackageManager()
    with pytest.raises(ValueError, match="No package names provided"):
        manager.install([])

    manager.install(["mlia", "pytest"])
    mock_check_output.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "mlia",
            "pytest",
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_py_package_manager_uninstall(mock_check_output: MagicMock) -> None:
    """Test package removal."""
    manager = PyPackageManager()
    with pytest.raises(ValueError, match="No package names provided"):
        manager.uninstall([])

    manager.uninstall(["mlia", "pytest"])
    mock_check_output.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "uninstall",
            "--yes",
            "mlia",
            "pytest",
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )
