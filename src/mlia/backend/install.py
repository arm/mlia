# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
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

from mlia.backend.repo import get_backend_repository
from mlia.utils.download import download
from mlia.utils.download import DownloadConfig
from mlia.utils.filesystem import all_files_exist
from mlia.utils.filesystem import temp_directory
from mlia.utils.filesystem import working_directory
from mlia.utils.py_manager import get_package_manager

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

    def __init__(
        self, name: str, description: str, dependencies: list[str] | None = None
    ) -> None:
        """Init the installation."""
        assert not dependencies or name not in dependencies, (
            f"Invalid backend configuration: Backend '{name}' has itself as a "
            "dependency! The backend source code needs to be fixed."
        )

        self.name = name
        self.description = description
        self.dependencies = dependencies if dependencies else []

    @property
    @abstractmethod
    def could_be_installed(self) -> bool:
        """Check if backend could be installed in current environment."""

    @property
    @abstractmethod
    def already_installed(self) -> bool:
        """Check if backend is already installed."""

    @abstractmethod
    def supports(self, install_type: InstallationType) -> bool:
        """Check if installation supports requested installation type."""

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
    settings: dict | None = None


PathChecker = Callable[[Path], Optional[BackendInfo]]
BackendInstaller = Callable[[bool, Path], Path]


class BackendInstallation(Installation):
    """Backend installation."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        description: str,
        fvp_dir_name: str,
        download_config: DownloadConfig | None,
        supported_platforms: list[str] | None,
        path_checker: PathChecker,
        backend_installer: BackendInstaller | None,
        dependencies: list[str] | None = None,
    ) -> None:
        """Init the backend installation."""
        super().__init__(name, description, dependencies)

        self.fvp_dir_name = fvp_dir_name
        self.download_config = download_config
        self.supported_platforms = supported_platforms
        self.path_checker = path_checker
        self.backend_installer = backend_installer

    @property
    def already_installed(self) -> bool:
        """Return true if backend already installed."""
        backend_repo = get_backend_repository()
        return backend_repo.is_backend_installed(self.name)

    @property
    def could_be_installed(self) -> bool:
        """Return true if backend could be installed."""
        return (
            not self.supported_platforms
            or platform.system() in self.supported_platforms
        )

    def supports(self, install_type: InstallationType) -> bool:
        """Return true if backends supported type of the installation."""
        if isinstance(install_type, DownloadAndInstall):
            return self.download_config is not None

        if isinstance(install_type, InstallFromPath):
            return self.path_checker(install_type.backend_path) is not None

        return False  # type: ignore

    def install(self, install_type: InstallationType) -> None:
        """Install the backend."""
        if isinstance(install_type, DownloadAndInstall):
            assert self.download_config is not None, "No artifact provided"

            self._download_and_install(
                self.download_config, install_type.eula_agreement
            )
        elif isinstance(install_type, InstallFromPath):
            backend_info = self.path_checker(install_type.backend_path)

            assert backend_info is not None, "Unable to resolve backend path"
            self._install_from(backend_info)
        else:
            raise RuntimeError(f"Unable to install {install_type}.")

    def _install_from(self, backend_info: BackendInfo) -> None:
        """Install backend from the directory."""
        backend_repo = get_backend_repository()

        if backend_info.copy_source:
            backend_repo.copy_backend(
                self.name,
                backend_info.backend_path,
                self.fvp_dir_name,
                backend_info.settings,
            )
        else:
            backend_repo.add_backend(
                self.name,
                backend_info.backend_path,
                backend_info.settings,
            )

    @staticmethod
    def _filter_tar_members(
        members: Iterable[tarfile.TarInfo], dst_dir: Path
    ) -> Iterable[tarfile.TarInfo]:
        """
        Make sure we only handle safe files from the tar file.

        To avoid traversal attacks we only allow files that are
        - relative paths, i.e. no absolute file paths
        - not including directory traversal sequences '..'
        """

        def check_rel(path: Path) -> None:
            if path.is_absolute():
                raise ValueError("Path is absolute, but must be relative.")

        def check_in_dir(path: Path) -> None:
            abs_path = (dst_dir / path).resolve()
            abs_path.relative_to(dst_dir)

        for member in members:
            try:
                path = Path(member.path)
                check_rel(path)
                check_in_dir(path)

                if member.islnk() or member.issym():
                    # Make sure we are only linking within the
                    # archive.
                    lnk = Path(member.linkname)
                    check_rel(lnk)
                    check_in_dir(lnk)

                yield member
            except ValueError as ex:
                logger.warning(
                    "File '%s' ignored while extracting: %s",
                    member.path,
                    ex,
                )

    def _download_and_install(self, cfg: DownloadConfig, eula_agrement: bool) -> None:
        """Download and install the backend."""
        with temp_directory() as tmpdir:
            try:
                dest = tmpdir / cfg.filename
                download(
                    dest=dest,
                    cfg=cfg,
                    show_progress=True,
                )

            except Exception as err:
                raise RuntimeError("Unable to download backend artifact.") from err

            with working_directory(tmpdir / "dist", create_dir=True) as dist_dir:
                with tarfile.open(dest) as archive:
                    # Filter files from the tarfile to avoid traversal attacks.
                    # Note: bandit is still putting out a low severity /
                    # low confidence warning despite the check
                    # From Python 3.9.17 on there is a built-in feature to fix
                    # this using the new argument filter="data", see
                    # https://docs.python.org/3.9/library/tarfile.html#tarfile.TarFile.extractall
                    logger.debug(
                        "Extracting downloaded artifact %s to %s.", dest, dist_dir
                    )
                    archive.extractall(  # nosec
                        dist_dir,
                        members=self._filter_tar_members(
                            archive.getmembers(), dist_dir
                        ),
                    )

                backend_path = dist_dir
                if self.backend_installer:
                    backend_path = self.backend_installer(eula_agrement, dist_dir)

                if self.path_checker(backend_path) is None:
                    raise ValueError("Downloaded artifact has invalid structure.")

                self.install(InstallFromPath(backend_path))

    def uninstall(self) -> None:
        """Uninstall the backend."""
        backend_repo = get_backend_repository()
        backend_repo.remove_backend(self.name)


class PackagePathChecker:
    """Package path checker."""

    def __init__(
        self,
        expected_files: list[str],
        backend_subfolder: str | None = None,
        settings: dict | None = None,
    ) -> None:
        """Init the path checker."""
        self.expected_files = expected_files
        self.backend_subfolder = backend_subfolder
        self.settings = settings

    def __call__(self, backend_path: Path) -> BackendInfo | None:
        """Check if directory contains all expected files."""
        resolved_paths = (backend_path / file for file in self.expected_files)
        if not all_files_exist(resolved_paths):
            return None

        actual_backend_path = backend_path
        if self.backend_subfolder:
            subfolder = backend_path / self.backend_subfolder

            if subfolder.is_dir():
                actual_backend_path = subfolder

        return BackendInfo(actual_backend_path, settings=self.settings)


class StaticPathChecker:
    """Static path checker."""

    def __init__(
        self,
        static_backend_path: Path,
        expected_files: list[str],
        copy_source: bool = False,
        settings: dict | None = None,
    ) -> None:
        """Init static path checker."""
        self.static_backend_path = static_backend_path
        self.expected_files = expected_files
        self.copy_source = copy_source
        self.settings = settings

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
            settings=self.settings,
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
        super().__init__(name, description)

        self._packages_to_install = packages_to_install
        self._packages_to_uninstall = packages_to_uninstall
        self._expected_packages = expected_packages

        self.package_manager = get_package_manager()

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
            raise ValueError(f"Unsupported installation type {install_type}.")

        self.package_manager.install(self._packages_to_install)

    def uninstall(self) -> None:
        """Uninstall the backend."""
        self.package_manager.uninstall(self._packages_to_uninstall)
