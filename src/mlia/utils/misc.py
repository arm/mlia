# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Various util functions."""
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from subprocess import CalledProcessError  # nosec
from subprocess import run  # nosec

from mlia.utils.filesystem import sha256


class MetadataError(Exception):
    """Metadata error."""


def yes(prompt: str) -> bool:
    """Return true if user confirms the action."""
    response = input(f"{prompt} [y/n]: ")
    return response in ["y", "Y"]


def get_pkg_version(pkg_name: str) -> str:
    """Return the version of python package."""
    try:
        pkg_version = metadata.version(pkg_name)
    except FileNotFoundError as exc:
        raise MetadataError(exc) from exc
    return pkg_version


def get_file_checksum(input_path: Path) -> str:
    """Return the checksum of the input model."""
    return sha256(input_path)


def is_docker_available() -> bool:
    """Check if we are running in a docker environment."""
    try:
        run(["docker", "version"], check=True, capture_output=True)  # nosec
        return True
    except (CalledProcessError, FileNotFoundError, PermissionError):
        return False


@lru_cache(maxsize=1)
def is_docker_available_cached() -> bool:
    """Cache result of is_docker_available()."""
    return is_docker_available()
