# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for installation process."""
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from mlia.utils.misc import yes


logger = logging.getLogger(__name__)


@dataclass
class InstallFromPath:
    """Installation from the local path."""

    backend_path: Path


@dataclass
class DownloadAndInstall:
    """Download and install."""

    eula_agreement: bool = True


InstallationType = Union[InstallFromPath, DownloadAndInstall]


class Installation(ABC):
    """Base class for the installation process of the backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return name of the backend."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return description of the backend."""

    @property
    @abstractmethod
    def could_be_installed(self) -> bool:
        """Return true if backend could be installed in current environment."""

    @property
    @abstractmethod
    def already_installed(self) -> bool:
        """Return true if backend is already installed."""

    @abstractmethod
    def supports(self, install_type: InstallationType) -> bool:
        """Return true if installation supports requested installation type."""

    @abstractmethod
    def install(self, install_type: InstallationType) -> None:
        """Install the backend."""


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

    def __init__(self, backend_name: Optional[str]) -> None:
        """Init filter."""
        self.backend_name = backend_name

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return not self.backend_name or installation.name == self.backend_name


class InstallationManager(ABC):
    """Helper class for managing installations."""

    @abstractmethod
    def install_from(self, backend_path: Path, backend_name: Optional[str]) -> None:
        """Install backend from the local directory."""

    @abstractmethod
    def download_and_install(
        self, backend_name: Optional[str], eula_agreement: bool
    ) -> None:
        """Download and install backends."""

    @abstractmethod
    def show_env_details(self) -> None:
        """Show environment details."""

    @abstractmethod
    def backend_installed(self, backend_name: str) -> bool:
        """Return true if requested backend installed."""


class InstallationFiltersMixin:
    """Mixin for filtering installation based on different conditions."""

    installations: List[Installation]

    def filter_by(self, *filters: InstallationFilter) -> List[Installation]:
        """Filter installations."""
        return [
            installation
            for installation in self.installations
            if all(filter_(installation) for filter_ in filters)
        ]

    def could_be_installed_from(
        self, backend_path: Path, backend_name: Optional[str]
    ) -> List[Installation]:
        """Return installations that could be installed from provided directory."""
        return self.filter_by(
            SupportsInstallTypeFilter(InstallFromPath(backend_path)),
            SearchByNameFilter(backend_name),
        )

    def could_be_downloaded_and_installed(
        self, backend_name: Optional[str] = None
    ) -> List[Installation]:
        """Return installations that could be downloaded and installed."""
        return self.filter_by(
            SupportsInstallTypeFilter(DownloadAndInstall()),
            SearchByNameFilter(backend_name),
            ReadyForInstallationFilter(),
        )

    def already_installed(
        self, backend_name: Optional[str] = None
    ) -> List[Installation]:
        """Return list of backends that are already installed."""
        return self.filter_by(
            AlreadyInstalledFilter(), SearchByNameFilter(backend_name)
        )

    def ready_for_installation(self) -> List[Installation]:
        """Return list of the backends that could be installed."""
        return self.filter_by(ReadyForInstallationFilter())


class DefaultInstallationManager(InstallationManager, InstallationFiltersMixin):
    """Interactive installation manager."""

    def __init__(
        self, installations: List[Installation], noninteractive: bool = False
    ) -> None:
        """Init the manager."""
        self.installations = installations
        self.noninteractive = noninteractive

    def choose_installation_for_path(
        self, backend_path: Path, backend_name: Optional[str]
    ) -> Optional[Installation]:
        """Check available installation and select one if possible."""
        installs = self.could_be_installed_from(backend_path, backend_name)

        if not installs:
            logger.info(
                "Unfortunatelly, it was not possible to automatically "
                "detect type of the installed FVP. "
                "Please, check provided path to the installed FVP."
            )
            return None

        if len(installs) != 1:
            names = ",".join((install.name for install in installs))
            logger.info(
                "Unable to correctly detect type of the installed FVP."
                "The following FVPs are detected %s. Installation skipped.",
                names,
            )
            return None

        installation = installs[0]
        if installation.already_installed:
            logger.info(
                "%s was found in %s, but it has been already installed.",
                installation.name,
                backend_path,
            )
            return None

        return installation

    def install_from(self, backend_path: Path, backend_name: Optional[str]) -> None:
        """Install from the provided directory."""
        installation = self.choose_installation_for_path(backend_path, backend_name)

        if not installation:
            return

        prompt = (
            f"{installation.name} was found in {backend_path}. "
            "Would you like to install it?"
        )
        self._install(installation, InstallFromPath(backend_path), prompt)

    def download_and_install(
        self, backend_name: Optional[str] = None, eula_agreement: bool = True
    ) -> None:
        """Download and install available backends."""
        installations = self.could_be_downloaded_and_installed(backend_name)

        if not installations:
            logger.info("No backends available for the installation.")
            return

        names = ",".join((installation.name for installation in installations))
        logger.info("Following backends are available for downloading: %s", names)

        for installation in installations:
            prompt = f"Would you like to download and install {installation.name}?"
            self._install(
                installation, DownloadAndInstall(eula_agreement=eula_agreement), prompt
            )

    def show_env_details(self) -> None:
        """Print current state of the execution environment."""
        if installed := self.already_installed():
            logger.info("Installed backends:\n")

            for installation in installed:
                logger.info("  - %s", installation.name)

        if could_be_installed := self.ready_for_installation():
            logger.info("Following backends could be installed:")

            for installation in could_be_installed:
                logger.info("  - %s", installation.name)

        if not installed and not could_be_installed:
            logger.info("No backends installed")

    def _install(
        self,
        installation: Installation,
        installation_type: InstallationType,
        prompt: str,
    ) -> None:
        proceed = self.noninteractive or yes(prompt)

        if proceed:
            installation.install(installation_type)
            logger.info("%s successfully installed.", installation.name)
        else:
            logger.info("%s installation canceled.", installation.name)

    def backend_installed(self, backend_name: str) -> bool:
        """Return true if requested backend installed."""
        installations = self.already_installed(backend_name)

        return len(installations) == 1
