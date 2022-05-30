# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use
"""Module for testing fs.py."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Union
from unittest.mock import MagicMock

import pytest

from aiet.utils.fs import get_resources
from aiet.utils.fs import read_file_as_bytearray
from aiet.utils.fs import read_file_as_string
from aiet.utils.fs import recreate_directory
from aiet.utils.fs import remove_directory
from aiet.utils.fs import remove_resource
from aiet.utils.fs import ResourceType
from aiet.utils.fs import valid_for_filename


@pytest.mark.parametrize(
    "resource_name,expected_path",
    [
        ("systems", does_not_raise()),
        ("applications", does_not_raise()),
        ("whaaat", pytest.raises(ResourceWarning)),
        (None, pytest.raises(ResourceWarning)),
    ],
)
def test_get_resources(resource_name: ResourceType, expected_path: Any) -> None:
    """Test get_resources() with multiple parameters."""
    with expected_path:
        resource_path = get_resources(resource_name)
        assert resource_path.exists()


def test_remove_resource_wrong_directory(
    monkeypatch: Any, test_applications_path: Path
) -> None:
    """Test removing resource with wrong directory."""
    mock_get_resources = MagicMock(return_value=test_applications_path)
    monkeypatch.setattr("aiet.utils.fs.get_resources", mock_get_resources)

    mock_shutil_rmtree = MagicMock()
    monkeypatch.setattr("aiet.utils.fs.shutil.rmtree", mock_shutil_rmtree)

    with pytest.raises(Exception, match="Resource .* does not exist"):
        remove_resource("unknown", "applications")
    mock_shutil_rmtree.assert_not_called()

    with pytest.raises(Exception, match="Wrong resource .*"):
        remove_resource("readme.txt", "applications")
    mock_shutil_rmtree.assert_not_called()


def test_remove_resource(monkeypatch: Any, test_applications_path: Path) -> None:
    """Test removing resource data."""
    mock_get_resources = MagicMock(return_value=test_applications_path)
    monkeypatch.setattr("aiet.utils.fs.get_resources", mock_get_resources)

    mock_shutil_rmtree = MagicMock()
    monkeypatch.setattr("aiet.utils.fs.shutil.rmtree", mock_shutil_rmtree)

    remove_resource("application1", "applications")
    mock_shutil_rmtree.assert_called_once()


def test_remove_directory(tmpdir: Any) -> None:
    """Test directory removal."""
    tmpdir_path = Path(tmpdir)
    tmpfile = tmpdir_path / "temp.txt"

    for item in [None, tmpfile]:
        with pytest.raises(Exception, match="No directory path provided"):
            remove_directory(item)

    newdir = tmpdir_path / "newdir"
    newdir.mkdir()

    assert newdir.is_dir()
    remove_directory(newdir)
    assert not newdir.exists()


def test_recreate_directory(tmpdir: Any) -> None:
    """Test directory recreation."""
    with pytest.raises(Exception, match="No directory path provided"):
        recreate_directory(None)

    tmpdir_path = Path(tmpdir)
    tmpfile = tmpdir_path / "temp.txt"
    tmpfile.touch()
    with pytest.raises(Exception, match="Path .* does exist and it is not a directory"):
        recreate_directory(tmpfile)

    newdir = tmpdir_path / "newdir"
    newdir.mkdir()
    newfile = newdir / "newfile"
    newfile.touch()
    assert list(newdir.iterdir()) == [newfile]
    recreate_directory(newdir)
    assert not list(newdir.iterdir())

    newdir2 = tmpdir_path / "newdir2"
    assert not newdir2.exists()
    recreate_directory(newdir2)
    assert newdir2.is_dir()


def write_to_file(
    write_directory: Any, write_mode: str, write_text: Union[str, bytes]
) -> Path:
    """Write some text to a temporary test file."""
    tmpdir_path = Path(write_directory)
    tmpfile = tmpdir_path / "file_name.txt"
    with open(tmpfile, write_mode) as file:  # pylint: disable=unspecified-encoding
        file.write(write_text)
    return tmpfile


class TestReadFileAsString:
    """Test read_file_as_string() function."""

    def test_returns_text_from_valid_file(self, tmpdir: Any) -> None:
        """Ensure the string written to a file read correctly."""
        file_path = write_to_file(tmpdir, "w", "hello")
        assert read_file_as_string(file_path) == "hello"

    def test_output_is_empty_string_when_input_file_non_existent(
        self, tmpdir: Any
    ) -> None:
        """Ensure empty string returned when reading from non-existent file."""
        file_path = Path(tmpdir / "non-existent.txt")
        assert read_file_as_string(file_path) == ""


class TestReadFileAsByteArray:
    """Test read_file_as_bytearray() function."""

    def test_returns_bytes_from_valid_file(self, tmpdir: Any) -> None:
        """Ensure the bytes written to a file read correctly."""
        file_path = write_to_file(tmpdir, "wb", b"hello bytes")
        assert read_file_as_bytearray(file_path) == b"hello bytes"

    def test_output_is_empty_bytearray_when_input_file_non_existent(
        self, tmpdir: Any
    ) -> None:
        """Ensure empty bytearray returned when reading from non-existent file."""
        file_path = Path(tmpdir / "non-existent.txt")
        assert read_file_as_bytearray(file_path) == bytearray()


@pytest.mark.parametrize(
    "value, replacement, expected_result",
    [
        ["", "", ""],
        ["123", "", "123"],
        ["123", "_", "123"],
        ["/some_folder/some_script.sh", "", "some_foldersome_script.sh"],
        ["/some_folder/some_script.sh", "_", "_some_folder_some_script.sh"],
        ["!;'some_name$%^!", "_", "___some_name____"],
    ],
)
def test_valid_for_filename(value: str, replacement: str, expected_result: str) -> None:
    """Test function valid_for_filename."""
    assert valid_for_filename(value, replacement) == expected_result
