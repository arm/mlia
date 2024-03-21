# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for common management functionality."""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from typing import Callable
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
from mlia.utils.download import DownloadConfig


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
    tmp_path: Path, backend_repo: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test methods of backend installation."""
    tmp_archive = tmp_path.joinpath("sample.tgz")
    sample_file = tmp_path.joinpath("sample.txt")
    sample_file.touch()

    with tarfile.open(tmp_archive, "w:gz") as archive:
        archive.add(sample_file)

    monkeypatch.setattr("mlia.backend.install.download", MagicMock())
    monkeypatch.setattr(
        "mlia.utils.download.DownloadConfig.filename",
        tmp_archive,
    )

    installation = BackendInstallation(
        "sample_backend",
        "Sample backend",
        "sample_backend",
        DownloadConfig(url="NOT_USED", sha256_hash="NOT_USED"),
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


def _gen_rel_file(dir_path: Path) -> Path:
    file_path = dir_path / "test.txt"
    if not file_path.exists():
        file_path.touch()
    return file_path


def _gen_abs_file(dir_path: Path) -> Path:
    return _gen_rel_file(dir_path).resolve()


def _gen_rel_sym(dir_path: Path) -> Path:
    file_path = _gen_rel_file(dir_path)
    lnk_path = dir_path / "symlink-rel"
    lnk_path.symlink_to(file_path.relative_to(dir_path))
    return lnk_path


def _gen_abs_sym(dir_path: Path) -> Path:
    file_path = _gen_abs_file(dir_path)
    lnk_path = dir_path / Path("symlink-abs")
    lnk_path.symlink_to(file_path)
    return lnk_path


def _gen_rel_lnk(dir_path: Path) -> Path:
    file_path = _gen_rel_file(dir_path)
    lnk_path = dir_path / "hardlink-rel"
    lnk_path.symlink_to(file_path.relative_to(dir_path))
    return lnk_path


def _gen_abs_lnk(dir_path: Path) -> Path:
    file_path = _gen_abs_file(dir_path)
    lnk_path = dir_path / Path("hardlink-abs")
    lnk_path.symlink_to(file_path)
    return lnk_path


@pytest.mark.parametrize(
    ("gen_file_func", "num_members_in", "num_members_out"),
    (
        (_gen_rel_file, 1, 1),
        (_gen_rel_sym, 1, 1),
        (_gen_abs_sym, 1, 0),
        (_gen_rel_lnk, 1, 1),
        (_gen_abs_lnk, 1, 0),
    ),
)
def test_filter_tar_members(
    gen_file_func: Callable[[Path], Path],
    num_members_in: int,
    num_members_out: int,
    tmp_path: Path,
) -> None:
    """Test function BackendInstallation._filter_tar_members()."""

    def create_tar(file_to_add: Path, archive_path: Path) -> None:
        with tarfile.open(archive_path, "w") as archive:
            archive.add(file_to_add, arcname=file_to_add, recursive=False)

    with tempfile.TemporaryDirectory() as tmp_dir_2:
        with pytest.raises(ValueError):
            Path(tmp_dir_2).relative_to(tmp_path)

        archive_path = Path(tmp_dir_2) / "test.tar.gz"
        file_path = gen_file_func(tmp_path)
        create_tar(file_path, archive_path)
        with tarfile.open(archive_path) as archive:
            orig_members = list(archive.getmembers())
            assert len(orig_members) == num_members_in
            filtered_members = list(
                # pylint: disable=protected-access
                BackendInstallation._filter_tar_members(orig_members, tmp_path)
            )
            assert len(filtered_members) == num_members_out
