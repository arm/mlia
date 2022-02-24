# Copyright (C) 2021-2022, Arm Ltd.
"""Tests for the filesystem module."""
import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import get_profile
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


def test_profiles_data_wrong_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test if profile data has wrong format."""
    wrong_profile_data = tmp_path / "bad.json"
    with open(wrong_profile_data, "w") as file:
        json.dump([], file)

    monkeypatch.setattr(
        "mlia.utils.filesystem.get_profiles_file",
        MagicMock(return_value=wrong_profile_data),
    )

    with pytest.raises(Exception, match="Profiles data format is not valid"):
        get_profiles_data()


def test_get_supported_profile_names() -> None:
    """Test profile names getter."""
    assert list(get_supported_profile_names()) == ["U55-256", "U55-128", "U65-512"]


def test_get_profile() -> None:
    """Test getting profile data."""
    assert get_profile("U55-256") == {
        "device": "ethos-u55",
        "mac": 256,
        "system_config": "Ethos_U55_High_End_Embedded",
        "memory_mode": "Shared_Sram",
    }

    with pytest.raises(Exception, match="Unable to find target profile unknown"):
        get_profile("unknown")


@pytest.mark.parametrize("raise_exception", [True, False])
def test_temp_file(raise_exception: bool) -> None:
    """Test temp_file context manager."""
    with contextlib.suppress(Exception):
        with temp_file() as tmp:
            tmp_path = Path(tmp)
            assert tmp_path.is_file()

            if raise_exception:
                raise Exception("Error!")

    assert not tmp_path.exists()
