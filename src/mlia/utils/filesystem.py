# Copyright 2021, Arm Ltd.
"""Utils related to file management."""
import importlib.resources as pkg_resources
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
from typing import Generator
from typing import Optional


def get_mlia_resources() -> Path:
    """Get the path to the resources."""
    ctx = pkg_resources.path("mlia", "resources")
    with ctx as resource_path:
        return resource_path


def get_vela_config() -> Path:
    """Get the path to the default vela config file."""
    return get_mlia_resources() / "vela/vela.ini"


@contextmanager
def temp_file(suffix: Optional[str] = None) -> Generator[str, None, None]:
    """Create temp file and remove it after."""
    _, tmp_file = mkstemp(suffix=suffix)

    yield tmp_file

    os.remove(tmp_file)
