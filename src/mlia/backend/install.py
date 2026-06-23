# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for installation process."""

from __future__ import annotations

import logging
import os
import platform
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from mlia.backend.repo import get_backend_repository
from mlia.backend.vendor import vendor_artifact_path
from mlia.utils.download import DownloadConfig, download
from mlia.utils.filesystem import all_files_exist, temp_directory
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


@dataclass
class InstallFromVendorPackage:
    """Installation from a vendor package."""


InstallationType = Union[InstallFromPath, DownloadAndInstall, InstallFromVendorPackage]


class Installation(ABC):
    """Base class for the installation process of the backends."""

    def __init__(
        self,
        name: str,
        description: str,
        dependencies: list[str] | None = None,
        requires_eula: bool = False,
    ) -> None:
        """Init the installation."""
        assert not dependencies or name not in dependencies, (
            f"Invalid backend configuration: Backend '{name}' has itself as a "
            "dependency! The backend source code needs to be fixed."
        )

        self.name = name
        self.description = description
        self.dependencies = dependencies if dependencies else []
        self.requires_eula = requires_eula

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

    def __eq__(self, other: object) -> bool:
        """Check equality with another Installation."""
        if isinstance(other, Installation):
            return (
                self.name == other.name
                and self.description == other.description
                and self.dependencies == other.dependencies
                and self.requires_eula == other.requires_eula
            )
        raise NotImplementedError


@dataclass
class BackendInfo:
    """Backend information."""

    backend_path: Path
    copy_source: bool = True
    settings: dict | None = None
    supporting_paths: list[tuple[Path, Path]] = field(default_factory=list)


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
        vendor_path: str | None = None,
        requires_eula: bool = False,
    ) -> None:
        """Init the backend installation."""
        super().__init__(name, description, dependencies, requires_eula)

        self.fvp_dir_name = fvp_dir_name
        self.download_config = download_config
        self.supported_platforms = supported_platforms
        self.path_checker = path_checker
        self.backend_installer = backend_installer
        self._vendor_path = vendor_path

    @property
    def vendor_path(self) -> Path | None:
        """Dynamically resolve the vendor path when accessed."""
        return (
            vendor_artifact_path(self._vendor_path)
            if self._vendor_path is not None
            else None
        )

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

        if isinstance(install_type, InstallFromVendorPackage):
            return self.vendor_path is not None
        return False  # type: ignore[unreachable]

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
        elif isinstance(install_type, InstallFromVendorPackage):
            resolved_path = self.vendor_path
            if resolved_path is None:
                raise RuntimeError("Vendor package is not available.")
            backend_info = self.path_checker(resolved_path)
            if backend_info is None:
                with temp_directory() as tmpdir:
                    backend_info = self._resolve_vendor_archive(resolved_path, tmpdir)
                    self._install_from(backend_info)
            else:
                self._install_from(backend_info)
        else:
            raise RuntimeError(f"Unable to install {install_type}.")

    def uninstall(self) -> None:
        """Uninstall the backend."""
        backend_repo = get_backend_repository()
        backend_repo.remove_backend(self.name)

    def _download_and_install(self, cfg: DownloadConfig, eula_agreement: bool) -> None:
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

            dist_dir = self._extract_archive_to_dist(dest, tmpdir)

            backend_path = dist_dir
            if self.backend_installer:
                backend_path = self.backend_installer(eula_agreement, dist_dir)

            if self.path_checker(backend_path) is None:
                raise ValueError("Downloaded artifact has invalid structure.")

            self.install(InstallFromPath(backend_path))

    def _resolve_vendor_archive(self, vendor_dir: Path, tmpdir: Path) -> BackendInfo:
        """Extract a vendored archive and return backend info."""
        archives = sorted(path.name for path in vendor_dir.glob("*.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(
                "Unable to resolve backend path from vendor archive. "
                f"Vendor directory: {vendor_dir}. "
                f"Found archives: {archives or 'none'}."
            )
        archive_path = vendor_dir / archives[0]
        dist_dir = self._extract_archive_to_dist(
            archive_path, tmpdir, log_prefix="vendor"
        )
        backend_path = dist_dir
        if self.backend_installer:
            backend_path = self.backend_installer(True, dist_dir)
        backend_info = self.path_checker(backend_path)
        if backend_info is None:
            raise RuntimeError(
                "Unable to resolve backend path from vendor archive. "
                f"Vendor directory: {vendor_dir}. "
                f"Archive: {archive_path.name}."
            )
        return backend_info

    def _extract_archive_to_dist(
        self, archive_path: Path, tmpdir: Path, log_prefix: str = "downloaded"
    ) -> Path:
        """Extract a tar.gz archive into a temp dist directory."""
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            logger.debug(
                "Extracting %s artifact %s to %s.",
                log_prefix,
                archive_path,
                dist_dir,
            )
            archive.extractall(
                dist_dir,
                members=_filter_tar_members(archive.getmembers(), dist_dir),
            )
        return dist_dir

    def _install_from(self, backend_info: BackendInfo) -> None:
        """Install backend from the directory."""
        backend_repo = get_backend_repository()
        logger.debug(
            "Installing %s in %s", backend_info.backend_path, backend_repo.repository
        )

        if backend_info.copy_source:
            backend_repo.copy_backend(
                self.name,
                backend_info.backend_path,
                self.fvp_dir_name,
                backend_info.settings,
                backend_info.supporting_paths,
            )
        else:
            backend_repo.add_backend(
                self.name,
                backend_info.backend_path,
                backend_info.settings,
            )


ARTIFACTORY_USERNAME_ENV_VAR = "MLIA_ARTIFACTORY_USERNAME"
ARTIFACTORY_PASSWORD_ENV_VAR = "MLIA_ARTIFACTORY_PASSWORD"  # nosec


def credentials_from_env(user_env: str, pw_env: str) -> tuple[str, str]:
    """Get the credentials from environment variables."""
    try:
        return (os.environ[user_env], os.environ[pw_env])
    except KeyError as ex:
        raise RuntimeError(
            "Failed to retrieve the credentials from environment variables."
            "Make sure your credentials are available as environment variables "
            f"'{user_env}' and '{pw_env}'."
        ) from ex


artifactory_credentials_from_env = partial(
    credentials_from_env, ARTIFACTORY_USERNAME_ENV_VAR, ARTIFACTORY_PASSWORD_ENV_VAR
)


def artifactory_credential_headers() -> dict[str, str]:
    """Get credentials from env vars and create HTTP headers for Artifactory."""
    username, password = artifactory_credentials_from_env()
    headers = {
        "Username": username,
        "X-JFrog-Art-Api": password,
    }
    return headers


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


def _resolve_supporting_paths(
    root_path: Path, supporting_subfolders: list[str]
) -> list[tuple[Path, Path]] | None:
    """Resolve configured supporting folders under root path."""
    supporting_paths: list[tuple[Path, Path]] = []
    root_path = root_path.resolve()
    for subfolder in supporting_subfolders:
        relative_path = Path(subfolder)
        supporting_path = (root_path / relative_path).resolve()

        if relative_path.is_absolute() or not supporting_path.is_dir():
            return None

        try:
            normalized_relative_path = supporting_path.relative_to(root_path)
        except ValueError:
            return None

        supporting_paths.append((supporting_path, normalized_relative_path))

    return supporting_paths


class PackagePathChecker:
    """Package path checker."""

    def __init__(
        self,
        expected_files: list[str],
        backend_subfolder: str | None = None,
        supporting_subfolders: list[str] | None = None,
        settings: dict | None = None,
    ) -> None:
        """Init the path checker."""
        self.expected_files = expected_files
        self.backend_subfolder = backend_subfolder
        self.supporting_subfolders = supporting_subfolders or []
        self.settings = settings

    def __call__(self, backend_path: Path) -> BackendInfo | None:
        """Check if directory contains all expected files."""
        resolved_paths = (backend_path / file for file in self.expected_files)
        if not all_files_exist(resolved_paths):
            return None

        actual_backend_path = backend_path
        if self.backend_subfolder:
            subfolder = backend_path / self.backend_subfolder

            if not subfolder.exists() or not subfolder.is_dir():
                logger.debug(
                    f"Backend subfolder '{self.backend_subfolder}'"
                    " not found in archive."
                )
                return None

            actual_backend_path = subfolder

        supporting_paths = _resolve_supporting_paths(
            backend_path, self.supporting_subfolders
        )
        if supporting_paths is None:
            return None

        return BackendInfo(
            actual_backend_path,
            settings=self.settings,
            supporting_paths=supporting_paths,
        )


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
        description: str,
        name: str,
        packages_to_install: list[str],
        packages_to_uninstall: list[str],
        expected_packages: list[str],
        download_config: DownloadConfig | None = None,
        vendor_path: str | None = None,
        requires_eula: bool = False,
    ) -> None:
        """Init the backend installation."""
        super().__init__(name, description, requires_eula=requires_eula)

        self.download_config = download_config
        self._packages_to_install = packages_to_install
        self._packages_to_uninstall = packages_to_uninstall
        self._expected_packages = expected_packages
        self._vendor_path = vendor_path

        self.package_manager = get_package_manager()

    @property
    def vendor_path(self) -> str | None:
        """Dynamically resolve the vendor path when accessed."""
        if self._vendor_path is None:
            return None
        resolved_path = vendor_artifact_path(self._vendor_path)
        if resolved_path is None:
            return None
        return " ".join(str(p) for p in resolved_path.glob("*.whl"))

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
        if isinstance(install_type, (DownloadAndInstall, InstallFromPath)):
            return True
        if isinstance(install_type, InstallFromVendorPackage):
            return self.vendor_path is not None
        return False  # type: ignore[unreachable]

    def install(self, install_type: InstallationType) -> None:
        """Install the backend."""
        if not self.supports(install_type):
            raise ValueError(
                f"Insufficient configuration for installation type {install_type}."
            )

        if self.download_config is not None and isinstance(
            install_type, DownloadAndInstall
        ):
            self._download_and_install(self.download_config)
            return

        if isinstance(install_type, InstallFromVendorPackage):
            if self.vendor_path is None:
                raise RuntimeError("Vendor package is not available.")
            self.package_manager.install(self.vendor_path.split())
            return

        if isinstance(install_type, InstallFromPath):
            self.package_manager.install([str(install_type.backend_path)])
            return

        self.package_manager.install(self._packages_to_install)

    def _download_and_install(self, cfg: DownloadConfig) -> None:
        """Download and install packages."""
        with temp_directory() as tmpdir:
            assert cfg.url.endswith(".whl"), "Only wheel files are supported."
            try:
                dest = tmpdir / cfg.filename
                download(
                    dest=dest,
                    cfg=cfg,
                    show_progress=True,
                )
            except Exception as err:
                raise RuntimeError("Unable to download wheel.") from err

            self.package_manager.install([str(dest)])

    def uninstall(self) -> None:
        """Uninstall the backend."""
        self.package_manager.uninstall(self._packages_to_uninstall)
