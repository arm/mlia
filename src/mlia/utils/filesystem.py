# Copyright (C) 2021-2022, Arm Ltd.
"""Utils related to file management."""
import importlib.resources as pkg_resources
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional


def get_mlia_resources() -> Path:
    """Get the path to the resources directory."""
    with pkg_resources.path("mlia", "resources") as resource_path:
        return resource_path


def get_vela_config() -> Path:
    """Get the path to the default vela config file."""
    return get_mlia_resources() / "vela/vela.ini"


def get_profiles_file() -> Path:
    """Get the EthosU profiles file."""
    return get_mlia_resources() / "profiles.json"


def get_profiles_data() -> Dict[str, Dict[str, Any]]:
    """Get the EthosU profile values as a dictionary."""
    with open(get_profiles_file()) as json_file:
        profiles = json.load(json_file)

        if not isinstance(profiles, dict):
            raise Exception("Profiles data format is not valid")

        return profiles


def get_profile(target: str) -> Dict[str, Any]:
    """Get settings for the provided target profile."""
    profiles = get_profiles_data()

    if target not in profiles:
        raise Exception(f"Unable to find target profile {target}")

    return profiles[target]


def get_supported_profile_names() -> List[str]:
    """Get the supported EthosU profile names."""
    return list(get_profiles_data().keys())


@contextmanager
def temp_file(suffix: Optional[str] = None) -> Generator[str, None, None]:
    """Create temp file and remove it after."""
    _, tmp_file = mkstemp(suffix=suffix)

    try:
        yield tmp_file
    finally:
        os.remove(tmp_file)
