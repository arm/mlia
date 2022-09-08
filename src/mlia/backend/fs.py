# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module to host all file system related functions."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

from mlia.utils.filesystem import get_mlia_resources

ResourceType = Literal["applications", "systems"]


def get_backend_resources() -> Path:
    """Get backend resources folder path."""
    return get_mlia_resources() / "backends"


def get_backends_path(name: ResourceType) -> Path:
    """Return the absolute path of the specified resource.

    It uses importlib to return resources packaged with MANIFEST.in.
    """
    if not name:
        raise ResourceWarning("Resource name is not provided")

    resource_path = get_backend_resources() / name
    if resource_path.is_dir():
        return resource_path

    raise ResourceWarning(f"Resource '{name}' not found.")


def copy_directory_content(source: Path, destination: Path) -> None:
    """Copy content of the source directory into destination directory."""
    for item in source.iterdir():
        src = source / item.name
        dest = destination / item.name

        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


def remove_resource(resource_directory: str, resource_type: ResourceType) -> None:
    """Remove resource data."""
    resources = get_backends_path(resource_type)

    resource_location = resources / resource_directory
    if not resource_location.exists():
        raise Exception(f"Resource {resource_directory} does not exist")

    if not resource_location.is_dir():
        raise Exception(f"Wrong resource {resource_directory}")

    shutil.rmtree(resource_location)


def remove_directory(directory_path: Path | None) -> None:
    """Remove directory."""
    if not directory_path or not directory_path.is_dir():
        raise Exception("No directory path provided")

    shutil.rmtree(directory_path)


def recreate_directory(directory_path: Path | None) -> None:
    """Recreate directory."""
    if not directory_path:
        raise Exception("No directory path provided")

    if directory_path.exists() and not directory_path.is_dir():
        raise Exception(
            f"Path {str(directory_path)} does exist and it is not a directory."
        )

    if directory_path.is_dir():
        remove_directory(directory_path)

    directory_path.mkdir()


def valid_for_filename(value: str, replacement: str = "") -> str:
    """Replace non alpha numeric characters."""
    return re.sub(r"[^\w.]", replacement, value, flags=re.ASCII)
