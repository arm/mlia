# Copyright 2021, Arm Ltd.
"""Tests for the filesystem module."""
from pathlib import Path

from mlia.utils.filesystem import temp_file


def test_temp_file() -> None:
    """Test temp_file context manager."""
    with temp_file() as tmp:
        tmp_path = Path(tmp)
        assert tmp_path.is_file()

    assert not tmp_path.exists()
