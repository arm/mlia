# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contain source related classes and functions."""
from __future__ import annotations

import os
import shutil
import tarfile
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from tarfile import TarFile

from mlia.backend.common import BACKEND_CONFIG_FILE
from mlia.backend.common import ConfigurationException
from mlia.backend.common import get_backend_config
from mlia.backend.common import is_backend_directory
from mlia.backend.common import load_config
from mlia.backend.config import BackendConfig
from mlia.backend.fs import copy_directory_content


class Source(ABC):
    """Source class."""

    @abstractmethod
    def name(self) -> str | None:
        """Get source name."""

    @abstractmethod
    def config(self) -> BackendConfig | None:
        """Get configuration file content."""

    @abstractmethod
    def install_into(self, destination: Path) -> None:
        """Install source into destination directory."""

    @abstractmethod
    def create_destination(self) -> bool:
        """Return True if destination folder should be created before installation."""


class DirectorySource(Source):
    """DirectorySource class."""

    def __init__(self, directory_path: Path) -> None:
        """Create the DirectorySource instance."""
        assert isinstance(directory_path, Path)
        self.directory_path = directory_path

    def name(self) -> str:
        """Return name of source."""
        return self.directory_path.name

    def config(self) -> BackendConfig | None:
        """Return configuration file content."""
        if not is_backend_directory(self.directory_path):
            raise ConfigurationException("No configuration file found")

        config_file = get_backend_config(self.directory_path)
        return load_config(config_file)

    def install_into(self, destination: Path) -> None:
        """Install source into destination directory."""
        if not destination.is_dir():
            raise ConfigurationException(f"Wrong destination {destination}.")

        if not self.directory_path.is_dir():
            raise ConfigurationException(
                f"Directory {self.directory_path} does not exist."
            )

        copy_directory_content(self.directory_path, destination)

    def create_destination(self) -> bool:
        """Return True if destination folder should be created before installation."""
        return True


class TarArchiveSource(Source):
    """TarArchiveSource class."""

    def __init__(self, archive_path: Path) -> None:
        """Create the TarArchiveSource class."""
        assert isinstance(archive_path, Path)
        self.archive_path = archive_path
        self._config: BackendConfig | None = None
        self._has_top_level_folder: bool | None = None
        self._name: str | None = None

    def _read_archive_content(self) -> None:
        """Read various information about archive."""
        # get source name from archive name (everything without extensions)
        extensions = "".join(self.archive_path.suffixes)
        self._name = self.archive_path.name.rstrip(extensions)

        if not self.archive_path.exists():
            return

        with self._open(self.archive_path) as archive:
            try:
                config_entry = archive.getmember(BACKEND_CONFIG_FILE)
                self._has_top_level_folder = False
            except KeyError as error_no_config:
                try:
                    archive_entries = archive.getnames()
                    entries_common_prefix = os.path.commonprefix(archive_entries)
                    top_level_dir = entries_common_prefix.rstrip("/")

                    if not top_level_dir:
                        raise RuntimeError(
                            "Archive has no top level directory"
                        ) from error_no_config

                    config_path = f"{top_level_dir}/{BACKEND_CONFIG_FILE}"

                    config_entry = archive.getmember(config_path)
                    self._has_top_level_folder = True
                    self._name = top_level_dir
                except (KeyError, RuntimeError) as error_no_root_dir_or_config:
                    raise ConfigurationException(
                        "No configuration file found"
                    ) from error_no_root_dir_or_config

            content = archive.extractfile(config_entry)
            self._config = load_config(content)

    def config(self) -> BackendConfig | None:
        """Return configuration file content."""
        if self._config is None:
            self._read_archive_content()

        return self._config

    def name(self) -> str | None:
        """Return name of the source."""
        if self._name is None:
            self._read_archive_content()

        return self._name

    def create_destination(self) -> bool:
        """Return True if destination folder must be created before installation."""
        if self._has_top_level_folder is None:
            self._read_archive_content()

        return not self._has_top_level_folder

    def install_into(self, destination: Path) -> None:
        """Install source into destination directory."""
        if not destination.is_dir():
            raise ConfigurationException(f"Wrong destination {destination}.")

        with self._open(self.archive_path) as archive:
            archive.extractall(destination)

    def _open(self, archive_path: Path) -> TarFile:
        """Open archive file."""
        if not archive_path.is_file():
            raise ConfigurationException(f"File {archive_path} does not exist.")

        if archive_path.name.endswith("tar.gz") or archive_path.name.endswith("tgz"):
            mode = "r:gz"
        else:
            raise ConfigurationException(f"Unsupported archive type {archive_path}.")

        # The returned TarFile object can be used as a context manager (using
        # 'with') by the calling instance.
        return tarfile.open(  # pylint: disable=consider-using-with
            self.archive_path, mode=mode
        )


def get_source(source_path: Path) -> TarArchiveSource | DirectorySource:
    """Return appropriate source instance based on provided source path."""
    if source_path.is_file():
        return TarArchiveSource(source_path)

    if source_path.is_dir():
        return DirectorySource(source_path)

    raise ConfigurationException(f"Unable to read {source_path}.")


def create_destination_and_install(source: Source, resource_path: Path) -> None:
    """Create destination directory and install source.

    This function is used for actual installation of system/backend New
    directory will be created inside :resource_path: if needed If for example
    archive contains top level folder then no need to create new directory
    """
    destination = resource_path
    create_destination = source.create_destination()

    if create_destination:
        name = source.name()
        if not name:
            raise ConfigurationException("Unable to get source name.")

        destination = resource_path / name
        destination.mkdir()
    try:
        source.install_into(destination)
    except Exception as error:
        if create_destination:
            shutil.rmtree(destination)
        raise error
