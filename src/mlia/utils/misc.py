# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Various util functions."""

from importlib import metadata
from pathlib import Path
from typing import Any

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
    """Retrun the checksum of the input model."""
    return sha256(input_path)


def merge(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively merges two dictionaries."""
    out = dict(a)
    for k in b:
        if k in out:
            if isinstance(b[k], dict) and isinstance(out[k], dict):
                out[k] = merge(out[k], b[k])
            else:
                out[k] = b[k]
        else:
            out[k] = b[k]

    return out
