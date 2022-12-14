# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utils related to file management."""
from __future__ import annotations

import hashlib
import importlib.resources as pkg_resources
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
from tempfile import TemporaryDirectory
from typing import Any
from typing import cast
from typing import Generator
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def get_mlia_resources() -> Path:
    """Get the path to the resources directory."""
    with pkg_resources.path("mlia", "__init__.py") as init_path:
        project_root = init_path.parent
        return project_root / "resources"


def get_vela_config() -> Path:
    """Get the path to the default Vela config file."""
    return get_mlia_resources() / "vela/vela.ini"


def get_mlia_target_profiles_dir() -> Path:
    """Get the profiles file."""
    return get_mlia_resources() / "target_profiles"


def get_profile_toml_file(target_profile: str | Path) -> str | Path:
    """Get the target profile toml file."""
    if not target_profile:
        raise Exception("Target profile is not provided")

    profile_toml_file = Path(get_mlia_target_profiles_dir() / f"{target_profile}.toml")
    if not profile_toml_file.is_file():
        profile_toml_file = Path(target_profile)

    if not profile_toml_file.exists():
        raise Exception(f"File not found: {profile_toml_file}.")
    return profile_toml_file


def get_profile(target_profile: str | Path) -> dict[str, Any]:
    """Get settings for the provided target profile."""
    if not target_profile:
        raise Exception("Target profile is not provided")

    toml_file = get_profile_toml_file(target_profile)

    with open(toml_file, "rb") as file:
        profile = tomllib.load(file)

    return cast(dict, profile)


def get_builtin_supported_profile_names() -> list[str]:
    """Return list of default profiles in the target profiles directory."""
    return sorted(
        [
            item.stem
            for item in get_mlia_target_profiles_dir().iterdir()
            if item.is_file() and item.suffix == ".toml"
        ]
    )


def get_target(target_profile: str | Path) -> str:
    """Return target for the provided target_profile."""
    profile = get_profile(target_profile)
    return cast(str, profile["target"])


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
    filepath: str | Path, hash_obj: hashlib._Hash  # pylint: disable=no-member
) -> str:
    """Return hex digest of the file."""
    for chunk in file_chunks(filepath):
        hash_obj.update(chunk)

    return hash_obj.hexdigest()


def sha256(filepath: Path) -> str:
    """Return SHA256 hash of the file."""
    return hexdigest(filepath, hashlib.sha256())


def all_files_exist(paths: Iterable[Path]) -> bool:
    """Check if all files are exist."""
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
