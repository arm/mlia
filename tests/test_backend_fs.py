# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for testing fs.py."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.fs import get_backends_path
from mlia.backend.fs import recreate_directory
from mlia.backend.fs import remove_directory
from mlia.backend.fs import remove_resource
from mlia.backend.fs import ResourceType
from mlia.backend.fs import valid_for_filename


@pytest.mark.parametrize(
    "resource_name,expected_path",
    [
        ("systems", does_not_raise()),
        ("applications", does_not_raise()),
        ("whaaat", pytest.raises(ResourceWarning)),
        (None, pytest.raises(ResourceWarning)),
    ],
)
def test_get_backends_path(resource_name: ResourceType, expected_path: Any) -> None:
    """Test get_resources() with multiple parameters."""
    with expected_path:
        resource_path = get_backends_path(resource_name)
        assert resource_path.exists()


def test_remove_resource_wrong_directory(
    monkeypatch: Any, test_applications_path: Path
) -> None:
    """Test removing resource with wrong directory."""
    mock_get_resources = MagicMock(return_value=test_applications_path)
    monkeypatch.setattr("mlia.backend.fs.get_backends_path", mock_get_resources)

    mock_shutil_rmtree = MagicMock()
    monkeypatch.setattr("mlia.backend.fs.shutil.rmtree", mock_shutil_rmtree)

    with pytest.raises(Exception, match="Resource .* does not exist"):
        remove_resource("unknown", "applications")
    mock_shutil_rmtree.assert_not_called()

    with pytest.raises(Exception, match="Wrong resource .*"):
        remove_resource("readme.txt", "applications")
    mock_shutil_rmtree.assert_not_called()


def test_remove_resource(monkeypatch: Any, test_applications_path: Path) -> None:
    """Test removing resource data."""
    mock_get_resources = MagicMock(return_value=test_applications_path)
    monkeypatch.setattr("mlia.backend.fs.get_backends_path", mock_get_resources)

    mock_shutil_rmtree = MagicMock()
    monkeypatch.setattr("mlia.backend.fs.shutil.rmtree", mock_shutil_rmtree)

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
    write_directory: Any, write_mode: str, write_text: str | bytes
) -> Path:
    """Write some text to a temporary test file."""
    tmpdir_path = Path(write_directory)
    tmpfile = tmpdir_path / "file_name.txt"
    with open(tmpfile, write_mode) as file:  # pylint: disable=unspecified-encoding
        file.write(write_text)
    return tmpfile


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
