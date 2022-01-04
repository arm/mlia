# Copyright 2021, Arm Ltd.
"""Tests for the filesystem module."""
from pathlib import Path

from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import get_profiles_data
from mlia.utils.filesystem import get_profiles_file
from mlia.utils.filesystem import get_supported_profile_names
from mlia.utils.filesystem import get_vela_config
from mlia.utils.filesystem import temp_file


def test_get_mlia_resources() -> None:
    """Test resources getter."""
    assert get_mlia_resources().is_dir()


def test_get_vela_config() -> None:
    """Test vela config files getter."""
    assert get_vela_config().is_file()
    assert get_vela_config().name == "vela.ini"


def test_profiles_file() -> None:
    """Test profiles file getter."""
    assert get_profiles_file().is_file()
    assert get_profiles_file().name == "profiles.json"


def test_profiles_data() -> None:
    """Test profiles data getter."""
    assert list(get_profiles_data().keys()) == ["U55-256", "U55-128", "U65-512"]


def test_get_supported_profile_names() -> None:
    """Test profile names getter."""
    assert list(get_supported_profile_names()) == ["U55-256", "U55-128", "U65-512"]


def test_temp_file() -> None:
    """Test temp_file context manager."""
    with temp_file() as tmp:
        tmp_path = Path(tmp)
        assert tmp_path.is_file()

    assert not tmp_path.exists()
