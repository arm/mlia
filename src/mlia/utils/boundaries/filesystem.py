# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified filesystem boundary gateways."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from typing import Generator, Iterable

from mlia.utils.logging import log_boundary_action

USER_ONLY_PERM_MASK = 0o700


def ensure_user_output_directory(path: Path) -> None:
    """Create a user-requested output directory when needed."""
    path.mkdir(parents=True, exist_ok=True)


def write_user_output_text(path: Path, content: str) -> None:
    """Write text to a user-requested output file."""
    path.write_text(content, encoding="utf-8")


def copy_user_output_file(source: Path, destination: str | Path) -> None:
    """Copy a file to a user-requested output location."""
    shutil.copy(source, destination)


def write_persistent_state_text(path: Path, content: str) -> None:
    """Write persistent state owned by MLIA."""
    path.write_text(content)


def write_persistent_state_text_with_notice(
    logger: logging.Logger, message: str, path: Path, content: str
) -> None:
    """Announce and write persistent state owned by MLIA."""
    log_boundary_action(logger, message)
    path.write_text(content)


def copy_persistent_state_tree(source: Path, destination: Path) -> None:
    """Copy a directory into MLIA-owned state."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def ensure_temporary_directory(path: Path) -> None:
    """Create a runtime temporary directory."""
    path.mkdir(parents=True, exist_ok=True)


def extract_temporary_archive(
    archive: tarfile.TarFile, destination: Path, members: Iterable[tarfile.TarInfo]
) -> None:
    """Extract an archive into a runtime temporary directory."""
    archive.extractall(destination, members=members)


def copy_all(*paths: Path, dest: Path) -> None:
    """Copy files or directories through an approved filesystem gateway."""
    dest.mkdir(exist_ok=True)

    for path in paths:
        if path.is_file():
            shutil.copy2(path, dest)
        elif path.is_dir():
            shutil.copytree(path, dest, dirs_exist_ok=True)


def recreate_directory(path: Path, mode: int = USER_ONLY_PERM_MASK) -> None:
    """Recreate a directory through an approved filesystem gateway."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
        shutil.rmtree(path)

    path.mkdir(exist_ok=True, mode=mode)


@contextmanager
def temporary_file(suffix: str | None = None) -> Generator[Path, None, None]:
    """Create a temporary file and remove it after use."""
    file_descriptor, tmp_file = mkstemp(suffix=suffix)
    os.close(file_descriptor)
    try:
        yield Path(tmp_file)
    finally:
        os.remove(tmp_file)


@contextmanager
def temporary_directory(suffix: str | None = None) -> Generator[Path, None, None]:
    """Create a temporary directory and remove it after use."""
    with TemporaryDirectory(suffix=suffix) as tmpdir:
        yield Path(tmpdir)


@contextmanager
def temporary_working_directory(
    working_dir: Path, create_dir: bool = False
) -> Generator[Path, None, None]:
    """Temporarily change the process working directory."""
    current_working_dir = Path.cwd()

    if create_dir:
        working_dir.mkdir()
    os.chdir(working_dir)

    try:
        yield working_dir
    finally:
        os.chdir(current_working_dir)


def copy_persistent_state_with_notice(
    logger: logging.Logger,
    message: str,
    *sources: Path,
    destination: Path,
) -> None:
    """Announce and copy files or directories into MLIA-owned state."""
    log_boundary_action(logger, message)
    destination.mkdir(exist_ok=True)

    for source in sources:
        if source.is_file():
            shutil.copy2(source, destination)
        elif source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)


def remove_persistent_state_with_notice(
    logger: logging.Logger, message: str, path: Path
) -> None:
    """Announce and remove an MLIA-owned directory tree."""
    log_boundary_action(logger, message)
    shutil.rmtree(path)


def create_persistent_state_directories_with_notice(
    logger: logging.Logger, message: str, *paths: Path
) -> None:
    """Announce and create MLIA-owned directories."""
    log_boundary_action(logger, message)
    for path in paths:
        path.mkdir()


def ensure_output_location_with_notice(
    logger: logging.Logger, message: str, path: Path
) -> None:
    """Announce and ensure an output location exists."""
    log_boundary_action(logger, message)
    path.mkdir(exist_ok=True, mode=USER_ONLY_PERM_MASK)


def recreate_output_directory_with_notice(
    logger: logging.Logger,
    message: str,
    path: Path,
    mode: int = USER_ONLY_PERM_MASK,
) -> None:
    """Announce and recreate a runtime output directory."""
    log_boundary_action(logger, message)

    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
        shutil.rmtree(path)

    path.mkdir(exist_ok=True, mode=mode)
