# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for installation process."""
from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable

from mlia.backend.config import BackendType
from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import Installation
from mlia.backend.install import InstallationType
from mlia.backend.install import InstallFromPath
from mlia.backend.registry import registry as backend_registry
from mlia.core.errors import ConfigurationError
from mlia.core.errors import InternalError
from mlia.utils.misc import yes


logger = logging.getLogger(__name__)

InstallationFilter = Callable[[Installation], bool]


class AlreadyInstalledFilter:
    """Filter for already installed backends."""

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return installation.already_installed


class ReadyForInstallationFilter:
    """Filter for ready to be installed backends."""

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return installation.could_be_installed and not installation.already_installed


class SupportsInstallTypeFilter:
    """Filter backends that support certain type of the installation."""

    def __init__(self, installation_type: InstallationType) -> None:
        """Init filter."""
        self.installation_type = installation_type

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return installation.supports(self.installation_type)


class SearchByNameFilter:
    """Filter installation by name."""

    def __init__(self, backend_name: str | None) -> None:
        """Init filter."""
        self.backend_name = backend_name

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return (
            not self.backend_name
            or installation.name.casefold() == self.backend_name.casefold()
        )


class InstallationManager(ABC):
    """Helper class for managing installations."""

    @abstractmethod
    def install_from(self, backend_path: Path, backend_name: str, force: bool) -> None:
        """Install backend from the local directory."""

    @abstractmethod
    def download_and_install(
        self, backend_name: str, eula_agreement: bool, force: bool
    ) -> None:
        """Download and install backends."""

    @abstractmethod
    def show_env_details(self) -> None:
        """Show environment details."""

    @abstractmethod
    def backend_installed(self, backend_name: str) -> bool:
        """Return true if requested backend installed."""

    @abstractmethod
    def uninstall(self, backend_name: str) -> None:
        """Delete the existing installation."""


class InstallationFiltersMixin:
    """Mixin for filtering installation based on different conditions."""

    installations: list[Installation]

    def filter_by(self, *filters: InstallationFilter) -> list[Installation]:
        """Filter installations."""
        return [
            installation
            for installation in self.installations
            if all(filter_(installation) for filter_ in filters)
        ]

    def find_by_name(self, backend_name: str) -> list[Installation]:
        """Return list of the backends filtered by name."""
        return self.filter_by(SearchByNameFilter(backend_name))

    def already_installed(self, backend_name: str = None) -> list[Installation]:
        """Return list of backends that are already installed."""
        return self.filter_by(
            AlreadyInstalledFilter(),
            SearchByNameFilter(backend_name),
        )

    def ready_for_installation(self) -> list[Installation]:
        """Return list of the backends that could be installed."""
        return self.filter_by(ReadyForInstallationFilter())


class DefaultInstallationManager(InstallationManager, InstallationFiltersMixin):
    """Interactive installation manager."""

    def __init__(
        self, installations: list[Installation], noninteractive: bool = False
    ) -> None:
        """Init the manager."""
        self.installations = installations
        self.noninteractive = noninteractive

    def _install(
        self,
        backend_name: str,
        install_type: InstallationType,
        prompt: Callable[[Installation], str],
        force: bool,
    ) -> None:
        """Check metadata and install backend."""
        installs = self.find_by_name(backend_name)

        if not installs:
            logger.info("Unknown backend '%s'.", backend_name)
            logger.info(
                "Please run command 'mlia-backend list' to get list of "
                "supported backend names."
            )

            return

        if len(installs) > 1:
            raise InternalError(f"More than one backend with name {backend_name} found")

        installation = installs[0]
        if not installation.supports(install_type):
            if isinstance(install_type, InstallFromPath):
                logger.info(
                    "Backend '%s' could not be installed using path '%s'.",
                    installation.name,
                    install_type.backend_path,
                )
                logger.info(
                    "Please check that '%s' is a valid path to the installed backend.",
                    install_type.backend_path,
                )
            else:
                logger.info(
                    "Backend '%s' could not be downloaded and installed",
                    installation.name,
                )
                logger.info(
                    "Please refer to the project's documentation for more details."
                )

            return

        if installation.already_installed and not force:
            logger.info("Backend '%s' is already installed.", installation.name)
            logger.info("Please, consider using --force option.")
            return

        proceed = self.noninteractive or yes(prompt(installation))
        if not proceed:
            logger.info("%s installation canceled.", installation.name)
            return

        if installation.already_installed and force:
            logger.info(
                "Force installing %s, so delete the existing "
                "installed backend first.",
                installation.name,
            )
            installation.uninstall()

        installation.install(install_type)
        logger.info("%s successfully installed.", installation.name)

    def install_from(
        self, backend_path: Path, backend_name: str, force: bool = False
    ) -> None:
        """Install from the provided directory."""

        def prompt(install: Installation) -> str:
            return (
                f"{install.name} was found in {backend_path}. "
                "Would you like to install it?"
            )

        install_type = InstallFromPath(backend_path)
        self._install(backend_name, install_type, prompt, force)

    def download_and_install(
        self, backend_name: str, eula_agreement: bool = True, force: bool = False
    ) -> None:
        """Download and install available backends."""

        def prompt(install: Installation) -> str:
            return f"Would you like to download and install {install.name}?"

        install_type = DownloadAndInstall(eula_agreement=eula_agreement)
        self._install(backend_name, install_type, prompt, force)

    def show_env_details(self) -> None:
        """Print current state of the execution environment."""
        if installed := self.already_installed():
            self._print_installation_list("Installed backends:", installed)

        if could_be_installed := self.ready_for_installation():
            self._print_installation_list(
                "Following backends could be installed:",
                could_be_installed,
                new_section=bool(installed),
            )

        if not installed and not could_be_installed:
            logger.info("No backends installed")

    @staticmethod
    def _print_installation_list(
        header: str, installations: list[Installation], new_section: bool = False
    ) -> None:
        """Print list of the installations."""
        logger.info("%s%s\n", "\n" if new_section else "", header)

        for installation in installations:
            logger.info("  - %s", installation.name)

    def uninstall(self, backend_name: str) -> None:
        """Uninstall the backend with name backend_name."""
        installations = self.already_installed(backend_name)

        if not installations:
            raise ConfigurationError(f"Backend '{backend_name}' is not installed")

        if len(installations) != 1:
            raise InternalError(
                f"More than one installed backend with name {backend_name} found"
            )

        installation = installations[0]
        installation.uninstall()

        logger.info("%s successfully uninstalled.", installation.name)

    def backend_installed(self, backend_name: str) -> bool:
        """Return true if requested backend installed."""
        installations = self.already_installed(backend_name)

        return len(installations) == 1


def get_installation_manager(noninteractive: bool = False) -> InstallationManager:
    """Return installation manager."""
    backends = [
        cfg.installation for cfg in backend_registry.items.values() if cfg.installation
    ]
    return DefaultInstallationManager(backends, noninteractive=noninteractive)


def get_available_backends() -> list[str]:
    """Return list of the available backends."""
    manager = get_installation_manager()
    available_backends = [
        backend
        for backend, cfg in backend_registry.items.items()
        if cfg.type == BackendType.BUILTIN or manager.backend_installed(backend)
    ]
    return available_backends
