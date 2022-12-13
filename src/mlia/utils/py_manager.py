# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Util functions for managing python packages."""
from __future__ import annotations

import logging
import subprocess  # nosec
import sys
from importlib.metadata import distribution
from importlib.metadata import PackageNotFoundError

from mlia.core.errors import InternalError


logger = logging.getLogger(__name__)


class PyPackageManager:
    """Python package manager."""

    @staticmethod
    def package_installed(pkg_name: str) -> bool:
        """Return true if package installed."""
        try:
            distribution(pkg_name)
        except PackageNotFoundError:
            return False

        return True

    def packages_installed(self, pkg_names: list[str]) -> bool:
        """Return true if all provided packages installed."""
        return all(self.package_installed(pkg) for pkg in pkg_names)

    def install(self, pkg_names: list[str]) -> None:
        """Install provided packages."""
        if not pkg_names:
            raise ValueError("No package names provided")

        self._execute_pip_cmd("install", pkg_names)

    def uninstall(self, pkg_names: list[str]) -> None:
        """Uninstall provided packages."""
        if not pkg_names:
            raise ValueError("No package names provided")

        self._execute_pip_cmd("uninstall", ["--yes", *pkg_names])

    @staticmethod
    def _execute_pip_cmd(subcommand: str, params: list[str]) -> None:
        """Execute pip command."""
        assert sys.executable, "Unable to launch pip command"

        try:
            output = subprocess.check_output(  # nosec
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    subcommand,
                    *params,
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = 0
        except subprocess.CalledProcessError as err:
            output = err.output
            returncode = err.returncode

        for line in output.splitlines():
            logger.debug(line.rstrip())

        if returncode != 0:
            raise InternalError("Unable to install python package")


def get_package_manager() -> PyPackageManager:
    """Get python packages manager."""
    return PyPackageManager()
