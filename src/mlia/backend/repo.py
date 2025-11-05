# SPDX-FileCopyrightText: Copyright 2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend repository.

Backend repository is responsible for managing backends
(apart from python package based) that have been installed
via command "mlia-backend".

Repository has associated directory (by default ~/.mlia) and
configuration file (by default ~/.mlia/mlia_config.json). In
configuration file repository keeps track of installed backends
and their settings. Backend settings could be used by MLIA for
correct instantiation of the backend.

If backend is removed then repository removes corresponding record
from configuration file along with backend files if needed.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from mlia.utils.filesystem import copy_all


class _ConfigFile:
    """Configuration file for backend repository."""

    def __init__(self, config_file: Path) -> None:
        """Init configuration file."""
        self.config_file = config_file
        self.config: dict = {"backends": []}

        if self.exists():
            content = self.config_file.read_text()
            self.config = json.loads(content)

    def exists(self) -> bool:
        """Check if configuration file exists."""
        return self.config_file.is_file()

    def save(self) -> None:
        """Save configuration."""
        content = json.dumps(self.config, indent=4)
        self.config_file.write_text(content)

    def add_backend(
        self,
        backend_name: str,
        settings: dict,
    ) -> None:
        """Add backend settings to configuration file."""
        item = {"name": backend_name, "settings": settings}
        self.config["backends"].append(item)

        self.save()

    def remove_backend(self, backend_name: str) -> None:
        """Remove backend settings."""
        backend = self._get_backend(backend_name)

        if backend:
            self.config["backends"].remove(backend)

        self.save()

    def backend_exists(self, backend_name: str) -> bool:
        """Check if backend exists in configuration file."""
        return self._get_backend(backend_name) is not None

    def _get_backend(self, backend_name: str) -> dict | None:
        """Find backend settings by backend name."""
        find_backend = (
            item for item in self.config["backends"] if item["name"] == backend_name
        )

        return next(find_backend, None)

    def get_backend_settings(self, backend_name: str) -> dict | None:
        """Get backend settings."""
        backend = self._get_backend(backend_name)

        return backend["settings"] if backend else None


class BackendRepository:
    """Repository for keeping track of the installed backends."""

    def __init__(
        self,
        repository: Path,
        config_filename: str = "mlia_config.json",
    ) -> None:
        """Init repository instance."""
        self.repository = repository
        self.config_file = _ConfigFile(repository / config_filename)

        self._init_repo()

    def copy_backend(
        self,
        backend_name: str,
        backend_path: Path,
        backend_dir_name: str,
        settings: dict | None = None,
    ) -> None:
        """Copy backend files into repository."""
        repo_backend_path = self._get_backend_path(backend_dir_name)

        if repo_backend_path.exists():
            raise RuntimeError(f"Unable to copy backend files for {backend_name}.")

        copy_all(backend_path, dest=repo_backend_path)

        settings = settings or {}
        settings["backend_dir"] = backend_dir_name

        self.config_file.add_backend(backend_name, settings)

    def add_backend(
        self,
        backend_name: str,
        backend_path: Path,
        settings: dict | None = None,
    ) -> None:
        """Add backend to repository."""
        if self.is_backend_installed(backend_name):
            raise RuntimeError(f"Backend {backend_name} already installed.")

        settings = settings or {}
        settings["backend_path"] = backend_path.absolute().as_posix()

        self.config_file.add_backend(backend_name, settings)

    def remove_backend(self, backend_name: str) -> None:
        """Remove backend from repository."""
        settings = self.config_file.get_backend_settings(backend_name)

        if not settings:
            raise RuntimeError(f"Backend {backend_name} is not installed.")

        if "backend_dir" in settings:
            repo_backend_path = self._get_backend_path(settings["backend_dir"])
            shutil.rmtree(repo_backend_path)

        self.config_file.remove_backend(backend_name)

    def is_backend_installed(self, backend_name: str) -> bool:
        """Check if backend is installed."""
        return self.config_file.backend_exists(backend_name)

    def get_backend_settings(self, backend_name: str) -> tuple[Path, dict]:
        """Return backend settings."""
        settings = self.config_file.get_backend_settings(backend_name)

        if not settings:
            raise RuntimeError(f"Backend {backend_name} is not installed.")

        if backend_dir := settings.get("backend_dir", None):
            return self._get_backend_path(backend_dir), settings

        if backend_path := settings.get("backend_path", None):
            return Path(backend_path), settings

        raise RuntimeError(f"Unable to resolve path of the backend {backend_name}.")

    def _get_backend_path(self, backend_dir_name: str) -> Path:
        """Return path to backend."""
        return self.repository.joinpath("backends", backend_dir_name)

    def _init_repo(self) -> None:
        """Init repository."""
        if self.repository.exists():
            if not self.config_file.exists():
                raise RuntimeError(
                    f"Directory {self.repository} could not be used as MLIA repository."
                )
        else:
            self.repository.mkdir()
            self.repository.joinpath("backends").mkdir()

            self.config_file.save()


def get_backend_repository(
    repository: Path | None = None,
) -> BackendRepository:
    """Return backend repository."""
    repo_path = repository
    if repo_path is None:
        venv_path = os.environ.get("VIRTUAL_ENV")
        if venv_path:
            repo_path = Path(venv_path).resolve().joinpath(".mlia")
        else:
            repo_path = Path.home() / ".mlia"
    return BackendRepository(repo_path)
