# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the filesystem module."""
import contextlib
from pathlib import Path

import pytest

from mlia.utils.filesystem import all_files_exist
from mlia.utils.filesystem import all_paths_valid
from mlia.utils.filesystem import copy_all
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import get_mlia_target_profiles_dir
from mlia.utils.filesystem import get_vela_config
from mlia.utils.filesystem import recreate_directory
from mlia.utils.filesystem import sha256
from mlia.utils.filesystem import temp_directory
from mlia.utils.filesystem import temp_file
from mlia.utils.filesystem import USER_ONLY_PERM_MASK
from mlia.utils.filesystem import working_directory
from tests.utils.common import check_expected_permissions


def test_get_mlia_resources() -> None:
    """Test resources getter."""
    assert get_mlia_resources().is_dir()


def test_get_vela_config() -> None:
    """Test Vela config files getter."""
    assert get_vela_config().is_file()
    assert get_vela_config().name == "vela.ini"


def test_get_mlia_target_profiles() -> None:
    """Test target profiles getter."""
    assert get_mlia_target_profiles_dir().is_dir()


@pytest.mark.parametrize("raise_exception", [True, False])
def test_temp_file(raise_exception: bool) -> None:
    """Test temp_file context manager."""
    with contextlib.suppress(RuntimeError):
        with temp_file() as tmp_path:
            assert tmp_path.is_file()

            if raise_exception:
                raise RuntimeError("Error!")

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


def test_recreate_directory(tmp_path: Path) -> None:
    """Test function recreate_directory."""
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()

    test_dir1 = sample_dir / "test_dir1"
    test_dir1.mkdir()

    test_file1 = test_dir1 / "sample_file1.txt"
    test_file1.touch()

    test_dir2 = sample_dir / "test_dir2"
    recreate_directory(test_dir2)

    assert test_dir1.is_dir()
    assert test_file1.is_file()
    assert test_dir2.is_dir()
    check_expected_permissions(test_dir2, USER_ONLY_PERM_MASK)

    recreate_directory(test_dir1)
    assert test_dir2.is_dir()
    assert not test_file1.exists()
    assert test_dir1.is_dir()
    check_expected_permissions(test_dir1, USER_ONLY_PERM_MASK)


def test_recreate_directory_wrong_path(tmp_path: Path) -> None:
    """Test that function should fail if provided path is not a directory."""
    sample_file = tmp_path / "sample_file.txt"
    sample_file.touch()

    with pytest.raises(ValueError, match=rf"Path {sample_file} is not a directory."):
        recreate_directory(sample_file)
