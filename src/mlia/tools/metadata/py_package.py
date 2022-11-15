# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for python package based installations."""
from __future__ import annotations

from mlia.tools.metadata.common import DownloadAndInstall
from mlia.tools.metadata.common import Installation
from mlia.tools.metadata.common import InstallationType
from mlia.utils.py_manager import get_package_manager


class PyPackageBackendInstallation(Installation):
    """Backend based on the python package."""

    def __init__(
        self,
        name: str,
        description: str,
        packages_to_install: list[str],
        packages_to_uninstall: list[str],
        expected_packages: list[str],
    ) -> None:
        """Init the backend installation."""
        self._name = name
        self._description = description
        self._packages_to_install = packages_to_install
        self._packages_to_uninstall = packages_to_uninstall
        self._expected_packages = expected_packages

        self.package_manager = get_package_manager()

    @property
    def name(self) -> str:
        """Return name of the backend."""
        return self._name

    @property
    def description(self) -> str:
        """Return description of the backend."""
        return self._description

    @property
    def could_be_installed(self) -> bool:
        """Check if backend could be installed."""
        return True

    @property
    def already_installed(self) -> bool:
        """Check if backend already installed."""
        return self.package_manager.packages_installed(self._expected_packages)

    def supports(self, install_type: InstallationType) -> bool:
        """Return true if installation supports requested installation type."""
        return isinstance(install_type, DownloadAndInstall)

    def install(self, install_type: InstallationType) -> None:
        """Install the backend."""
        if not self.supports(install_type):
            raise Exception(f"Unsupported installation type {install_type}")

        self.package_manager.install(self._packages_to_install)

    def uninstall(self) -> None:
        """Uninstall the backend."""
        self.package_manager.uninstall(self._packages_to_uninstall)


def get_tosa_backend_installation() -> Installation:
    """Get TOSA backend installation."""
    return PyPackageBackendInstallation(
        name="tosa-checker",
        description="Tool to check if a ML model is compatible "
        "with the TOSA specification",
        packages_to_install=["mlia[tosa]"],
        packages_to_uninstall=["tosa-checker"],
        expected_packages=["tosa-checker"],
    )


def get_pypackage_backend_installations() -> list[Installation]:
    """Return list of the backend installations based on python packages."""
    return [
        get_tosa_backend_installation(),
    ]
