# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Various util functions."""
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from subprocess import CalledProcessError  # nosec
from subprocess import run  # nosec
from typing import Any
from typing import Dict
from typing import List
from typing import Union

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


def list_to_dict(list_mapping: list, key_field: Any) -> Union[Dict, List]:
    """Convert a list to a dict with key key_field."""
    if key_field:
        dict_mapping = {}
        for list_map in list_mapping:
            try:
                new_key = list_map.pop(key_field)
                dict_mapping[new_key] = list_map
            except KeyError as exc:
                raise KeyError("The key_field isn't present in all dicts.") from exc
        return dict_mapping

    return list_mapping


def dict_to_list(dict_mapping: dict, key_field: Any) -> list:
    """Convert a dict to a list of dicts."""
    output = []
    for key, value in dict_mapping.items():
        out = {**value}
        out[key_field] = key
        output.append(out)

    return output
