# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for common management functionality."""
from __future__ import annotations

import tarfile
from pathlib import Path
from unittest.mock import ANY
from unittest.mock import MagicMock

import pytest

from mlia.backend.install import BackendInfo
from mlia.backend.install import BackendInstallation
from mlia.backend.install import CompoundPathChecker
from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import InstallFromPath
from mlia.backend.install import PackagePathChecker
from mlia.backend.install import StaticPathChecker
from mlia.backend.repo import BackendRepository


@pytest.fixture(name="backend_repo")
def mock_backend_repo(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock backend repository."""
    mock = MagicMock(spec=BackendRepository)
    monkeypatch.setattr("mlia.backend.install.get_backend_repository", lambda: mock)

    return mock


def test_wrong_install_type() -> None:
    """Test that installation should fail for wrong install type."""
    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        None,
        None,
        lambda path: None,
        None,
    )

    assert not installation.supports("some_path")  # type: ignore

    with pytest.raises(Exception):
        installation.install("some_path")  # type: ignore


@pytest.mark.parametrize(
    "supported_platforms, expected_result",
    [
        [None, True],
        [["UNKNOWN"], False],
    ],
)
def test_backend_could_be_installed(
    supported_platforms: list[str] | None, expected_result: bool
) -> None:
    """Test method could_be_installed."""
    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        None,
        supported_platforms,
        lambda path: None,
        None,
    )

    assert installation.could_be_installed == expected_result


@pytest.mark.parametrize("copy_source", [True, False])
def test_backend_installation_from_path(
    tmp_path: Path, backend_repo: MagicMock, copy_source: bool
) -> None:
    """Test methods of backend installation."""
    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        None,
        None,
        lambda path: BackendInfo(path, copy_source=copy_source),
        None,
    )

    assert installation.supports(InstallFromPath(tmp_path))
    assert not installation.supports(DownloadAndInstall())

    installation.install(InstallFromPath(tmp_path))

    if copy_source:
        backend_repo.copy_backend.assert_called_with(
            "sample_backend", tmp_path, "sample_backend", None
        )
        backend_repo.add_backend.assert_not_called()
    else:
        backend_repo.copy_backend.assert_not_called()
        backend_repo.add_backend.assert_called_with("sample_backend", tmp_path, None)


def test_backend_installation_download_and_install(
    tmp_path: Path, backend_repo: MagicMock
) -> None:
    """Test methods of backend installation."""
    download_artifact_mock = MagicMock()

    tmp_archive = tmp_path.joinpath("sample.tgz")
    sample_file = tmp_path.joinpath("sample.txt")
    sample_file.touch()

    with tarfile.open(tmp_archive, "w:gz") as archive:
        archive.add(sample_file)

    download_artifact_mock.download_to.return_value = tmp_archive

    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        download_artifact_mock,
        None,
        lambda path: BackendInfo(path, copy_source=False),
        lambda eula_agreement, path: path,
    )

    assert installation.supports(DownloadAndInstall())
    installation.install(DownloadAndInstall())

    backend_repo.add_backend.assert_called_with("sample_backend", ANY, None)


def test_backend_installation_unable_to_download() -> None:
    """Test that installation should fail when downloading fails."""
    download_artifact_mock = MagicMock()
    download_artifact_mock.download_to.side_effect = Exception("Download error")

    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        download_artifact_mock,
        None,
        lambda path: BackendInfo(path, copy_source=False),
        lambda eula_agreement, path: path,
    )

    with pytest.raises(Exception, match="Unable to download backend artifact"):
        installation.install(DownloadAndInstall())


def test_static_path_checker(tmp_path: Path) -> None:
    """Test for StaticPathChecker."""
    checker1 = StaticPathChecker(tmp_path, [])
    assert checker1(tmp_path) == BackendInfo(tmp_path, copy_source=False)

    checker2 = StaticPathChecker(tmp_path / "dist", [])
    assert checker2(tmp_path) is None

    checker3 = StaticPathChecker(tmp_path, ["sample.txt"])

    assert checker3(tmp_path) is None

    sample_file = tmp_path.joinpath("sample.txt")
    sample_file.touch()

    assert checker3(tmp_path) == BackendInfo(tmp_path, copy_source=False)


def test_compound_path_checker(tmp_path: Path) -> None:
    """Test for CompoundPathChecker."""
    static_checker = StaticPathChecker(tmp_path, [])
    compound_checker = CompoundPathChecker(static_checker)

    assert compound_checker(tmp_path) == BackendInfo(tmp_path, copy_source=False)


def test_package_path_checker(tmp_path: Path) -> None:
    """Test PackagePathChecker."""
    sample_dir = tmp_path.joinpath("sample")
    sample_dir.mkdir()

    checker1 = PackagePathChecker([], "sample")
    assert checker1(tmp_path) == BackendInfo(tmp_path / "sample")

    checker2 = PackagePathChecker(["sample.txt"], "sample")
    assert checker2(tmp_path) is None


def test_backend_installation_uninstall(backend_repo: MagicMock) -> None:
    """Test backend removing process."""
    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        None,
        None,
        lambda path: None,
        None,
    )

    installation.uninstall()
    backend_repo.remove_backend.assert_called_with("sample_backend")
