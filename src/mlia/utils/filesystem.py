# Copyright 2021, Arm Ltd.
"""Utils related to file management."""
import os
from contextlib import contextmanager
from tempfile import mkstemp
from typing import Generator


@contextmanager
def temp_file() -> Generator[str, None, None]:
    """Create temp file and remove it after."""
    _, tmp_file = mkstemp()

    yield tmp_file

    os.remove(tmp_file)
