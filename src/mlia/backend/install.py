# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for installation process."""
from __future__ import annotations

import logging
import platform
import tarfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Union

from mlia.backend.executor.runner import BackendRunner
from mlia.backend.executor.system import remove_system
from mlia.utils.download import DownloadArtifact
from mlia.utils.filesystem import all_files_exist
from mlia.utils.filesystem import all_paths_valid
from mlia.utils.filesystem import copy_all
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import temp_directory
from mlia.utils.filesystem import working_directory
from mlia.utils.py_manager import get_package_manager

logger = logging.getLogger(__name__)


# Mapping backend -> device_type -> system_name
_SUPPORTED_SYSTEMS = {
    "Corstone-300": {
        "Ethos-U55": "Corstone-300: Cortex-M55+Ethos-U55",
        "Ethos-U65": "Corstone-300: Cortex-M55+Ethos-U65",
        "ethos-u55": "Corstone-300: Cortex-M55+Ethos-U55",
        "ethos-u65": "Corstone-300: Cortex-M55+Ethos-U65",
    },
    "Corstone-310": {
        "Ethos-U55": "Corstone-310: Cortex-M85+Ethos-U55",
        "Ethos-U65": "Corstone-310: Cortex-M85+Ethos-U65",
        "ethos-u55": "Corstone-310: Cortex-M85+Ethos-U55",
        "ethos-u65": "Corstone-310: Cortex-M85+Ethos-U65",
    },
}

# Mapping system_name -> application
_SYSTEM_TO_APP_MAP = {
    "Corstone-300: Cortex-M55+Ethos-U55": "Generic Inference Runner: Ethos-U55",
    "Corstone-300: Cortex-M55+Ethos-U65": "Generic Inference Runner: Ethos-U65",
    "Corstone-310: Cortex-M85+Ethos-U55": "Generic Inference Runner: Ethos-U55",
    "Corstone-310: Cortex-M85+Ethos-U65": "Generic Inference Runner: Ethos-U65",
}


def get_system_name(backend: str, device_type: str) -> str:
    """Get the system name for the given backend and device type."""
    return _SUPPORTED_SYSTEMS[backend][device_type]


def get_application_name(system_name: str) -> str:
    """Get application name for the provided system name."""
    return _SYSTEM_TO_APP_MAP[system_name]


def get_all_system_names(backend: str) -> list[str]:
    """Get all systems supported by the backend."""
    return list(_SUPPORTED_SYSTEMS.get(backend, {}).values())


def get_all_application_names(backend: str) -> list[str]:
    """Get all applications supported by the backend."""
    app_set = {_SYSTEM_TO_APP_MAP[sys] for sys in get_all_system_names(backend)}
    return list(app_set)


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

    @abstractmethod
    def uninstall(self) -> None:
        """Uninstall the backend."""


@dataclass
class BackendInfo:
    """Backend information."""

    backend_path: Path
    copy_source: bool = True
    system_config: str | None = None


PathChecker = Callable[[Path], Optional[BackendInfo]]
BackendInstaller = Callable[[bool, Path], Path]


class BackendMetadata:
    """Backend installation metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        system_config: str,
        apps_resources: list[str],
        fvp_dir_name: str,
        download_artifact: DownloadArtifact | None,
        supported_platforms: list[str] | None = None,
    ) -> None:
        """
        Initialize BackendMetadata.

        Members expected_systems and expected_apps are filled automatically.
        """
        self.name = name
        self.description = description
        self.system_config = system_config
        self.apps_resources = apps_resources
        self.fvp_dir_name = fvp_dir_name
        self.download_artifact = download_artifact
        self.supported_platforms = supported_platforms

        self.expected_systems = get_all_system_names(name)
        self.expected_apps = get_all_application_names(name)

    @property
    def expected_resources(self) -> Iterable[Path]:
        """Return list of expected resources."""
        resources = [self.system_config, *self.apps_resources]

        return (get_mlia_resources() / resource for resource in resources)

    @property
    def supported_platform(self) -> bool:
        """Return true if current platform supported."""
        if not self.supported_platforms:
            return True

        return platform.system() in self.supported_platforms


class BackendInstallation(Installation):
    """Backend installation."""

    def __init__(
        self,
        backend_runner: BackendRunner,
        metadata: BackendMetadata,
        path_checker: PathChecker,
        backend_installer: BackendInstaller | None,
    ) -> None:
        """Init the backend installation."""
        self.backend_runner = backend_runner
        self.metadata = metadata
        self.path_checker = path_checker
        self.backend_installer = backend_installer

    @property
    def name(self) -> str:
        """Return name of the backend."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Return description of the backend."""
        return self.metadata.description

    @property
    def already_installed(self) -> bool:
        """Return true if backend already installed."""
        return self.backend_runner.all_installed(
            self.metadata.expected_systems, self.metadata.expected_apps
        )

    @property
    def could_be_installed(self) -> bool:
        """Return true if backend could be installed."""
        if not self.metadata.supported_platform:
            return False

        return all_paths_valid(self.metadata.expected_resources)

    def supports(self, install_type: InstallationType) -> bool:
        """Return true if backends supported type of the installation."""
        if isinstance(install_type, DownloadAndInstall):
            return self.metadata.download_artifact is not None

        if isinstance(install_type, InstallFromPath):
            return self.path_checker(install_type.backend_path) is not None

        return False  # type: ignore

    def install(self, install_type: InstallationType) -> None:
        """Install the backend."""
        if isinstance(install_type, DownloadAndInstall):
            download_artifact = self.metadata.download_artifact
            assert download_artifact is not None, "No artifact provided"

            self.download_and_install(download_artifact, install_type.eula_agreement)
        elif isinstance(install_type, InstallFromPath):
            backend_path = self.path_checker(install_type.backend_path)
            assert backend_path is not None, "Unable to resolve backend path"

            self.install_from(backend_path)
        else:
            raise Exception(f"Unable to install {install_type}")

    def install_from(self, backend_info: BackendInfo) -> None:
        """Install backend from the directory."""
        mlia_resources = get_mlia_resources()

        with temp_directory() as tmpdir:
            fvp_dist_dir = tmpdir / self.metadata.fvp_dir_name

            system_config = self.metadata.system_config
            if backend_info.system_config:
                system_config = backend_info.system_config

            resources_to_copy = [mlia_resources / system_config]
            if backend_info.copy_source:
                resources_to_copy.append(backend_info.backend_path)

            copy_all(*resources_to_copy, dest=fvp_dist_dir)

            self.backend_runner.install_system(fvp_dist_dir)

        for app in self.metadata.apps_resources:
            self.backend_runner.install_application(mlia_resources / app)

    def download_and_install(
        self, download_artifact: DownloadArtifact, eula_agrement: bool
    ) -> None:
        """Download and install the backend."""
        with temp_directory() as tmpdir:
            try:
                downloaded_to = download_artifact.download_to(tmpdir)
            except Exception as err:
                raise Exception("Unable to download backend artifact") from err

            with working_directory(tmpdir / "dist", create_dir=True) as dist_dir:
                with tarfile.open(downloaded_to) as archive:
                    archive.extractall(dist_dir)

                assert self.backend_installer, (
                    f"Backend '{self.metadata.name}' does not support "
                    "download and installation."
                )
                backend_path = self.backend_installer(eula_agrement, dist_dir)
                if self.path_checker(backend_path) is None:
                    raise Exception("Downloaded artifact has invalid structure")

                self.install(InstallFromPath(backend_path))

    def uninstall(self) -> None:
        """Uninstall the backend."""
        remove_system(self.metadata.fvp_dir_name)


class PackagePathChecker:
    """Package path checker."""

    def __init__(
        self, expected_files: list[str], backend_subfolder: str | None = None
    ) -> None:
        """Init the path checker."""
        self.expected_files = expected_files
        self.backend_subfolder = backend_subfolder

    def __call__(self, backend_path: Path) -> BackendInfo | None:
        """Check if directory contains all expected files."""
        resolved_paths = (backend_path / file for file in self.expected_files)
        if not all_files_exist(resolved_paths):
            return None

        if self.backend_subfolder:
            subfolder = backend_path / self.backend_subfolder

            if not subfolder.is_dir():
                return None

            return BackendInfo(subfolder)

        return BackendInfo(backend_path)


class StaticPathChecker:
    """Static path checker."""

    def __init__(
        self,
        static_backend_path: Path,
        expected_files: list[str],
        copy_source: bool = False,
        system_config: str | None = None,
    ) -> None:
        """Init static path checker."""
        self.static_backend_path = static_backend_path
        self.expected_files = expected_files
        self.copy_source = copy_source
        self.system_config = system_config

    def __call__(self, backend_path: Path) -> BackendInfo | None:
        """Check if directory equals static backend path with all expected files."""
        if backend_path != self.static_backend_path:
            return None

        resolved_paths = (backend_path / file for file in self.expected_files)
        if not all_files_exist(resolved_paths):
            return None

        return BackendInfo(
            backend_path,
            copy_source=self.copy_source,
            system_config=self.system_config,
        )


class CompoundPathChecker:
    """Compound path checker."""

    def __init__(self, *path_checkers: PathChecker) -> None:
        """Init compound path checker."""
        self.path_checkers = path_checkers

    def __call__(self, backend_path: Path) -> BackendInfo | None:
        """Iterate over checkers and return first non empty backend info."""
        first_resolved_backend_info = (
            backend_info
            for path_checker in self.path_checkers
            if (backend_info := path_checker(backend_path)) is not None
        )

        return next(first_resolved_backend_info, None)


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
