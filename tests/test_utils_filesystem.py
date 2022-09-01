# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the filesystem module."""
import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.utils.filesystem import all_files_exist
from mlia.utils.filesystem import all_paths_valid
from mlia.utils.filesystem import copy_all
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import get_profile
from mlia.utils.filesystem import get_profiles_data
from mlia.utils.filesystem import get_profiles_file
from mlia.utils.filesystem import get_supported_profile_names
from mlia.utils.filesystem import get_vela_config
from mlia.utils.filesystem import sha256
from mlia.utils.filesystem import temp_directory
from mlia.utils.filesystem import temp_file
from mlia.utils.filesystem import working_directory


def test_get_mlia_resources() -> None:
    """Test resources getter."""
    assert get_mlia_resources().is_dir()


def test_get_vela_config() -> None:
    """Test Vela config files getter."""
    assert get_vela_config().is_file()
    assert get_vela_config().name == "vela.ini"


def test_profiles_file() -> None:
    """Test profiles file getter."""
    assert get_profiles_file().is_file()
    assert get_profiles_file().name == "profiles.json"


def test_profiles_data() -> None:
    """Test profiles data getter."""
    assert list(get_profiles_data().keys()) == [
        "ethos-u55-256",
        "ethos-u55-128",
        "ethos-u65-512",
        "ethos-u65-256",
        "tosa",
    ]


def test_profiles_data_wrong_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test if profile data has wrong format."""
    wrong_profile_data = tmp_path / "bad.json"
    with open(wrong_profile_data, "w", encoding="utf-8") as file:
        json.dump([], file)

    monkeypatch.setattr(
        "mlia.utils.filesystem.get_profiles_file",
        MagicMock(return_value=wrong_profile_data),
    )

    with pytest.raises(Exception, match="Profiles data format is not valid"):
        get_profiles_data()


def test_get_supported_profile_names() -> None:
    """Test profile names getter."""
    assert list(get_supported_profile_names()) == [
        "ethos-u55-256",
        "ethos-u55-128",
        "ethos-u65-512",
        "ethos-u65-256",
        "tosa",
    ]


def test_get_profile() -> None:
    """Test getting profile data."""
    assert get_profile("ethos-u55-256") == {
        "target": "ethos-u55",
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
        with temp_file() as tmp_path:
            assert tmp_path.is_file()

            if raise_exception:
                raise Exception("Error!")

    assert not tmp_path.exists()


def test_sha256(tmp_path: Path) -> None:
    """Test getting sha256 hash."""
    sample = tmp_path / "sample.txt"

    with open(sample, "w", encoding="utf-8") as file:
        file.write("123")

    expected_hash = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
    assert sha256(sample) == expected_hash


def test_temp_dir_context_manager() -> None:
    """Test context manager for temporary directories."""
    with temp_directory() as tmpdir:
        assert isinstance(tmpdir, Path)
        assert tmpdir.is_dir()

    assert not tmpdir.exists()


def test_all_files_exist(tmp_path: Path) -> None:
    """Test function all_files_exist."""
    sample1 = tmp_path / "sample1.txt"
    sample1.touch()

    sample2 = tmp_path / "sample2.txt"
    sample2.touch()

    sample3 = tmp_path / "sample3.txt"

    assert all_files_exist([sample1, sample2]) is True
    assert all_files_exist([sample1, sample2, sample3]) is False


def test_all_paths_valid(tmp_path: Path) -> None:
    """Test function all_paths_valid."""
    sample = tmp_path / "sample.txt"
    sample.touch()

    sample_dir = tmp_path / "sample_dir"
    sample_dir.mkdir()

    unknown = tmp_path / "unknown.txt"

    assert all_paths_valid([sample, sample_dir]) is True
    assert all_paths_valid([sample, sample_dir, unknown]) is False


def test_copy_all(tmp_path: Path) -> None:
    """Test function copy_all."""
    sample = tmp_path / "sample1.txt"
    sample.touch()

    sample_dir = tmp_path / "sample_dir"
    sample_dir.mkdir()

    sample_nested_file = sample_dir / "sample_nested.txt"
    sample_nested_file.touch()

    dest_dir = tmp_path / "dest"
    copy_all(sample, sample_dir, dest=dest_dir)

    assert (dest_dir / sample.name).is_file()
    assert (dest_dir / sample_nested_file.name).is_file()


@pytest.mark.parametrize(
    "should_exist, create_dir",
    [
        [True, False],
        [False, True],
    ],
)
def test_working_directory_context_manager(
    tmp_path: Path, should_exist: bool, create_dir: bool
) -> None:
    """Test working_directory context manager."""
    prev_wd = Path.cwd()

    working_dir = tmp_path / "work_dir"
    if should_exist:
        working_dir.mkdir()

    with working_directory(working_dir, create_dir=create_dir) as current_working_dir:
        assert current_working_dir.is_dir()
        assert Path.cwd() == current_working_dir

    assert Path.cwd() == prev_wd
