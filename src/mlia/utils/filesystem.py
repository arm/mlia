# Copyright 2021, Arm Ltd.
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
from typing import Iterable
from typing import Optional


def get_mlia_resources() -> Path:
    """Get the path to the resources directory."""
    ctx = pkg_resources.path("mlia", "resources")
    with ctx as resource_path:
        return resource_path


def get_vela_config() -> Path:
    """Get the path to the default vela config file."""
    return get_mlia_resources() / "vela/vela.ini"


def get_profiles_file() -> Path:
    """Get the EthosU profiles file."""
    return get_mlia_resources() / "profiles.json"


def get_profiles_data() -> Dict[str, Any]:
    """Get the EthosU profile values as a dictionary."""
    with open(get_profiles_file()) as json_file:
        profile = json.load(json_file)
        if isinstance(profile, dict):
            return profile
        raise Exception("Profiles data format is not valid.")


def get_supported_profile_names() -> Iterable[str]:
    """Get the supported EthosU profile names."""
    return get_profiles_data().keys()


@contextmanager
def temp_file(suffix: Optional[str] = None) -> Generator[str, None, None]:
    """Create temp file and remove it after."""
    _, tmp_file = mkstemp(suffix=suffix)

    yield tmp_file

    os.remove(tmp_file)
