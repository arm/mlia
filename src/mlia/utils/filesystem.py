# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utils related to file management."""

from __future__ import annotations

import hashlib
import importlib.resources as pkg_resources
import os
import shutil
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from types import ModuleType
from typing import Generator, Iterable

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

USER_ONLY_PERM_MASK = 0o700


def get_mlia_resource_dirs() -> list[Path]:
    """Get all available resources directories from installed MLIA packages."""
    try:
        mlia_module: ModuleType | None = import_module("mlia")
    except Exception:  # pragma: no cover - defensive guard
        mlia_module = None

    resource_dirs: list[Path] = []
    if mlia_module is not None:
        for base in mlia_module.__path__:
            candidate = Path(base) / "resources"
            if candidate.exists():
                resource_dirs.append(candidate)

    # Fallback: inspect installed distributions for packaged resources
    try:
        for dist in metadata.distributions():
            files = dist.files or []
            has_resources = any(
                str(file).startswith("mlia/resources/") for file in files
            )
            if not has_resources:
                continue
            candidate = Path(dist.locate_file("mlia/resources"))
            if candidate.exists():
                resource_dirs.append(candidate)
    except Exception:  # pragma: no cover - defensive guard
        pass

    # Fallback: scan sys.path entries for namespace resources
    try:
        for path_entry in sys.path:
            candidate = Path(path_entry) / "mlia" / "resources"
            if candidate.exists():
                resource_dirs.append(candidate)
    except Exception:  # pragma: no cover - defensive guard
        pass

    # De-duplicate while preserving order
    deduped: list[Path] = []
    seen: set[Path] = set()
    for item in resource_dirs:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    resource_dirs = deduped

    if resource_dirs:
        return resource_dirs

    with pkg_resources.path("mlia", "__init__.py") as init_path:
        project_root = init_path.parent
        return [project_root / "resources"]


def get_mlia_resources() -> Path:
    """Get the path to the primary resources directory."""
    return get_mlia_resource_dirs()[0]


def get_vela_config() -> Path:
    """Get the path to the default Vela config file."""
    for resources_dir in get_mlia_resource_dirs():
        candidate = resources_dir / "vela/vela.ini"
        if candidate.exists():
            return candidate
    return get_mlia_resources() / "vela/vela.ini"


def get_mlia_target_profiles_dir() -> Path:
    """Get the profiles file."""
    return get_mlia_resources() / "target_profiles"


def get_mlia_target_optimization_dir() -> Path:
    """Get the profiles file."""
    return get_mlia_resources() / "optimization_profiles"


@contextmanager
def temp_file(suffix: str | None = None) -> Generator[Path, None, None]:
    """Create temp file and remove it after."""
    _, tmp_file = mkstemp(suffix=suffix)

    try:
        yield Path(tmp_file)
    finally:
        os.remove(tmp_file)


@contextmanager
def temp_directory(suffix: str | None = None) -> Generator[Path, None, None]:
    """Create temp directory and remove it after."""
    with TemporaryDirectory(suffix=suffix) as tmpdir:
        yield Path(tmpdir)


def file_chunks(
    filepath: str | Path, chunk_size: int = 4096
) -> Generator[bytes, None, None]:
    """Return sequence of the file chunks."""
    with open(filepath, "rb") as file:
        while data := file.read(chunk_size):
            yield data


def hexdigest(
    filepath: str | Path,
    hash_obj: hashlib._Hash,  # pylint: disable=no-member
) -> str:
    """Return hex digest of the file."""
    for chunk in file_chunks(filepath):
        hash_obj.update(chunk)

    return hash_obj.hexdigest()


def sha256(filepath: Path) -> str:
    """Return SHA256 hash of the file."""
    return hexdigest(filepath, hashlib.sha256())


def all_files_exist(paths: Iterable[Path]) -> bool:
    """Check if all files exist."""
    return all(item.is_file() for item in paths)


def all_paths_valid(paths: Iterable[Path]) -> bool:
    """Check if all paths are valid."""
    return all(item.exists() for item in paths)


def copy_all(*paths: Path, dest: Path) -> None:
    """Copy files/directories into destination folder."""
    dest.mkdir(exist_ok=True)

    for path in paths:
        if path.is_file():
            shutil.copy2(path, dest)

        if path.is_dir():
            shutil.copytree(path, dest, dirs_exist_ok=True)


def recreate_directory(dir_path: Path, mode: int = USER_ONLY_PERM_MASK) -> None:
    """Recreate directory."""
    if dir_path.exists():
        if not dir_path.is_dir():
            raise ValueError(f"Path {dir_path} is not a directory.")

        shutil.rmtree(dir_path)

    dir_path.mkdir(exist_ok=True, mode=mode)


@contextmanager
def working_directory(
    working_dir: Path, create_dir: bool = False
) -> Generator[Path, None, None]:
    """Temporary change working directory."""
    current_working_dir = Path.cwd()

    if create_dir:
        working_dir.mkdir()

    os.chdir(working_dir)

    try:
        yield working_dir
    finally:
        os.chdir(current_working_dir)
