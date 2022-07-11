# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use
"""Tests for the source backend module."""
from collections import Counter
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from mlia.backend.common import ConfigurationException
from mlia.backend.source import create_destination_and_install
from mlia.backend.source import DirectorySource
from mlia.backend.source import get_source
from mlia.backend.source import TarArchiveSource


def test_create_destination_and_install(test_systems_path: Path, tmpdir: Any) -> None:
    """Test create_destination_and_install function."""
    system_directory = test_systems_path / "system1"

    dir_source = DirectorySource(system_directory)
    resources = Path(tmpdir)
    create_destination_and_install(dir_source, resources)
    assert (resources / "system1").is_dir()


@patch("mlia.backend.source.DirectorySource.create_destination", return_value=False)
def test_create_destination_and_install_if_dest_creation_not_required(
    mock_ds_create_destination: Any, tmpdir: Any
) -> None:
    """Test create_destination_and_install function."""
    dir_source = DirectorySource(Path("unknown"))
    resources = Path(tmpdir)
    with pytest.raises(Exception):
        create_destination_and_install(dir_source, resources)

    mock_ds_create_destination.assert_called_once()


def test_create_destination_and_install_if_installation_fails(tmpdir: Any) -> None:
    """Test create_destination_and_install function if installation fails."""
    dir_source = DirectorySource(Path("unknown"))
    resources = Path(tmpdir)
    with pytest.raises(Exception, match="Directory .* does not exist"):
        create_destination_and_install(dir_source, resources)
    assert not (resources / "unknown").exists()
    assert resources.exists()


def test_create_destination_and_install_if_name_is_empty() -> None:
    """Test create_destination_and_install function fails if source name is empty."""
    source = MagicMock()
    source.create_destination.return_value = True
    source.name.return_value = None

    with pytest.raises(Exception, match="Unable to get source name"):
        create_destination_and_install(source, Path("some_path"))

    source.install_into.assert_not_called()


@pytest.mark.parametrize(
    "source_path, expected_class, expected_error",
    [
        (
            Path("backends/applications/application1/"),
            DirectorySource,
            does_not_raise(),
        ),
        (
            Path("archives/applications/application1.tar.gz"),
            TarArchiveSource,
            does_not_raise(),
        ),
        (
            Path("doesnt/exist"),
            None,
            pytest.raises(
                ConfigurationException, match="Unable to read .*doesnt/exist"
            ),
        ),
    ],
)
def test_get_source(
    source_path: Path,
    expected_class: Any,
    expected_error: Any,
    test_resources_path: Path,
) -> None:
    """Test get_source function."""
    with expected_error:
        full_source_path = test_resources_path / source_path
        source = get_source(full_source_path)
        assert isinstance(source, expected_class)


class TestDirectorySource:
    """Test DirectorySource class."""

    @pytest.mark.parametrize(
        "directory, name",
        [
            (Path("/some/path/some_system"), "some_system"),
            (Path("some_system"), "some_system"),
        ],
    )
    def test_name(self, directory: Path, name: str) -> None:
        """Test getting source name."""
        assert DirectorySource(directory).name() == name

    def test_install_into(self, test_systems_path: Path, tmpdir: Any) -> None:
        """Test install directory into destination."""
        system_directory = test_systems_path / "system1"

        dir_source = DirectorySource(system_directory)
        with pytest.raises(Exception, match="Wrong destination .*"):
            dir_source.install_into(Path("unknown_destination"))

        tmpdir_path = Path(tmpdir)
        dir_source.install_into(tmpdir_path)
        source_files = [f.name for f in system_directory.iterdir()]
        dest_files = [f.name for f in tmpdir_path.iterdir()]
        assert Counter(source_files) == Counter(dest_files)

    def test_install_into_unknown_source_directory(self, tmpdir: Any) -> None:
        """Test install system from unknown directory."""
        with pytest.raises(Exception, match="Directory .* does not exist"):
            DirectorySource(Path("unknown_directory")).install_into(Path(tmpdir))


class TestTarArchiveSource:
    """Test TarArchiveSource class."""

    @pytest.mark.parametrize(
        "archive, name",
        [
            (Path("some_archive.tgz"), "some_archive"),
            (Path("some_archive.tar.gz"), "some_archive"),
            (Path("some_archive"), "some_archive"),
            ("archives/systems/system1.tar.gz", "system1"),
            ("archives/systems/system1_dir.tar.gz", "system1"),
        ],
    )
    def test_name(self, test_resources_path: Path, archive: Path, name: str) -> None:
        """Test getting source name."""
        assert TarArchiveSource(test_resources_path / archive).name() == name

    def test_install_into(self, test_resources_path: Path, tmpdir: Any) -> None:
        """Test install archive into destination."""
        system_archive = test_resources_path / "archives/systems/system1.tar.gz"

        tar_source = TarArchiveSource(system_archive)
        with pytest.raises(Exception, match="Wrong destination .*"):
            tar_source.install_into(Path("unknown_destination"))

        tmpdir_path = Path(tmpdir)
        tar_source.install_into(tmpdir_path)
        source_files = [
            "backend-config.json.license",
            "backend-config.json",
            "system_artifact",
        ]
        dest_files = [f.name for f in tmpdir_path.iterdir()]
        assert Counter(source_files) == Counter(dest_files)

    def test_install_into_unknown_source_archive(self, tmpdir: Any) -> None:
        """Test install unknown source archive."""
        with pytest.raises(Exception, match="File .* does not exist"):
            TarArchiveSource(Path("unknown.tar.gz")).install_into(Path(tmpdir))

    def test_install_into_unsupported_source_archive(self, tmpdir: Any) -> None:
        """Test install unsupported file type."""
        plain_text_file = Path(tmpdir) / "test_file"
        plain_text_file.write_text("Not a system config")

        with pytest.raises(Exception, match="Unsupported archive type .*"):
            TarArchiveSource(plain_text_file).install_into(Path(tmpdir))

    def test_lazy_property_init(self, test_resources_path: Path) -> None:
        """Test that class properties initialized correctly."""
        system_archive = test_resources_path / "archives/systems/system1.tar.gz"

        tar_source = TarArchiveSource(system_archive)
        assert tar_source.name() == "system1"
        assert tar_source.config() is not None
        assert tar_source.create_destination()

        tar_source = TarArchiveSource(system_archive)
        assert tar_source.config() is not None
        assert tar_source.create_destination()
        assert tar_source.name() == "system1"

    def test_create_destination_property(self, test_resources_path: Path) -> None:
        """Test create_destination property filled correctly for different archives."""
        system_archive1 = test_resources_path / "archives/systems/system1.tar.gz"
        system_archive2 = test_resources_path / "archives/systems/system1_dir.tar.gz"

        assert TarArchiveSource(system_archive1).create_destination()
        assert not TarArchiveSource(system_archive2).create_destination()
