# SPDX-FileCopyrightText: Copyright 2022, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for python package manager."""

import logging
import subprocess  # nosec
import sys
from unittest.mock import MagicMock

import pytest

from mlia.core.errors import InternalError
from mlia.utils.py_manager import PyPackageManager, get_package_manager


def test_get_package_manager() -> None:
    """Test function get_package_manager."""
    manager = get_package_manager()
    assert isinstance(manager, PyPackageManager)


@pytest.fixture(name="mock_check_output")
def mock_check_output_fixture(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock check_call function."""
    mock_check_output = MagicMock()

    monkeypatch.setattr(
        "mlia.utils.boundaries.process.subprocess.check_output", mock_check_output
    )

    return mock_check_output


def test_py_package_manager_metadata() -> None:
    """Test getting package status."""
    manager = PyPackageManager()
    assert manager.package_installed("pytest")
    assert manager.packages_installed(["pytest", "mlia"])


def test_py_package_manager_install(
    mock_check_output: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Test package installation."""
    manager = PyPackageManager()
    with pytest.raises(ValueError, match="No package names provided"):
        manager.install([])

    with caplog.at_level(logging.INFO, logger="mlia.utils.py_manager"):
        manager.install(["mlia", "pytest"])
    record = next(
        record for record in caplog.records if "Running pip install" in record.message
    )
    assert getattr(record, "boundary_action", False) is True
    assert record.getMessage() == (
        f"Running pip install using Python interpreter '{sys.executable}'."
    )
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


def test_py_package_manager_does_not_log_credentials(
    mock_check_output: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Pip notices do not include credential-bearing requirement arguments."""
    requirement = "package @ https://user:password@example.test/pkg?token=secret"

    with caplog.at_level(logging.INFO, logger="mlia.utils.py_manager"):
        PyPackageManager().install([requirement])

    assert "Running pip install" in caplog.text
    assert requirement not in caplog.text
    assert "user" not in caplog.text
    assert "password" not in caplog.text
    assert "secret" not in caplog.text
    mock_check_output.assert_called_once()


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


def test_py_package_manager_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test PyPackageManager handling CalledProcessError."""
    monkeypatch.setattr(
        "subprocess.check_output",
        MagicMock(
            side_effect=subprocess.CalledProcessError(
                cmd="", output="Test output\nTest output 2\n", returncode=10
            )
        ),
    )

    manager = PyPackageManager()
    with pytest.raises(InternalError, match=r"^Unable to install python package"):
        manager.install(["mlia", "pytest"])
