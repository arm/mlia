# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for python package based installations."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.tools.metadata.common import DownloadAndInstall
from mlia.tools.metadata.common import InstallFromPath
from mlia.tools.metadata.py_package import get_pypackage_backend_installations
from mlia.tools.metadata.py_package import get_tosa_backend_installation
from mlia.tools.metadata.py_package import PyPackageBackendInstallation


def test_get_pypackage_backends() -> None:
    """Test function get_pypackage_backends."""
    backend_installs = get_pypackage_backend_installations()

    assert isinstance(backend_installs, list)
    assert len(backend_installs) == 1

    tosa_installation = backend_installs[0]
    assert isinstance(tosa_installation, PyPackageBackendInstallation)


def test_get_tosa_backend_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test function get_tosa_backend_installation."""
    mock_package_manager = MagicMock()
    monkeypatch.setattr(
        "mlia.tools.metadata.py_package.get_package_manager",
        lambda: mock_package_manager,
    )

    tosa_installation = get_tosa_backend_installation()

    assert isinstance(tosa_installation, PyPackageBackendInstallation)
    assert tosa_installation.name == "tosa-checker"
    assert (
        tosa_installation.description
        == "Tool to check if a ML model is compatible with the TOSA specification"
    )
    assert tosa_installation.could_be_installed
    assert tosa_installation.supports(DownloadAndInstall())
    assert not tosa_installation.supports(InstallFromPath(tmp_path))

    mock_package_manager.packages_installed.return_value = True
    assert tosa_installation.already_installed
    mock_package_manager.packages_installed.assert_called_once_with(["tosa-checker"])

    with pytest.raises(Exception, match=r"Unsupported installation type.*"):
        tosa_installation.install(InstallFromPath(tmp_path))

    mock_package_manager.install.assert_not_called()

    tosa_installation.install(DownloadAndInstall())
    mock_package_manager.install.assert_called_once_with(["mlia[tosa]"])

    tosa_installation.uninstall()
    mock_package_manager.uninstall.assert_called_once_with(["tosa-checker"])
