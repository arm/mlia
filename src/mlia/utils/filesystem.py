# Copyright 2021, Arm Ltd.
"""Utils related to file management."""
import os
from contextlib import contextmanager
from tempfile import mkstemp
from typing import Generator
from typing import Optional


@contextmanager
def temp_file(suffix: Optional[str] = None) -> Generator[str, None, None]:
    """Create temp file and remove it after."""
    _, tmp_file = mkstemp(suffix=suffix)

    yield tmp_file

    os.remove(tmp_file)
