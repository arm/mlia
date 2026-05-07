# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend installation from vendored archives."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from mlia.backend.install import (
    BackendInstallation,
    InstallFromVendorPackage,
    PackagePathChecker,
)
from mlia.backend.repo import get_backend_repository


def _create_tar_with_file(archive_path: Path, filename: str) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    file_path = archive_path.parent / filename
    file_path.write_text("data", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(file_path, arcname=filename)
    file_path.unlink()


def test_install_from_vendor_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Install backend when vendored artifacts contain a tar.gz archive."""
    repo_path = tmp_path / "repo"
    repo = get_backend_repository(repo_path)
    monkeypatch.setattr("mlia.backend.install.get_backend_repository", lambda: repo)

    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    _create_tar_with_file(archive_path, "tool")

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )

    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(expected_files=["tool"]),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )

    installation.install(InstallFromVendorPackage())

    backend_dir = repo_path / "backends" / "sample-backend"
    assert backend_dir.is_dir()
    assert (backend_dir / "tool").is_file()


def test_backend_installation_accepts_requires_eula() -> None:
    """Backend installation should expose EULA requirements at construction."""
    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(expected_files=["tool"]),
        backend_installer=None,
        requires_eula=True,
    )

    assert installation.requires_eula is True


def _make_installation(
    monkeypatch: pytest.MonkeyPatch, repo_path: Path, vendor_dir: Path
) -> BackendInstallation:
    repo = get_backend_repository(repo_path)
    monkeypatch.setattr("mlia.backend.install.get_backend_repository", lambda: repo)
    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )
    return BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(expected_files=["tool"]),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )


def test_install_from_vendor_archive_missing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when no vendor archives are present."""
    repo_path = tmp_path / "repo"
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    vendor_dir.mkdir(parents=True)

    installation = _make_installation(monkeypatch, repo_path, vendor_dir)

    with pytest.raises(RuntimeError, match="Found archives: none"):
        installation.install(InstallFromVendorPackage())


def test_install_from_vendor_archive_multiple_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when multiple vendor archives are present."""
    repo_path = tmp_path / "repo"
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    _create_tar_with_file(vendor_dir / "a.tar.gz", "tool")
    _create_tar_with_file(vendor_dir / "b.tar.gz", "tool")

    installation = _make_installation(monkeypatch, repo_path, vendor_dir)

    with pytest.raises(RuntimeError, match="Found archives: .*a.tar.gz.*b.tar.gz"):
        installation.install(InstallFromVendorPackage())


def test_install_from_vendor_archive_invalid_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when archive does not contain expected files."""
    repo_path = tmp_path / "repo"
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    _create_tar_with_file(vendor_dir / "sample-backend.tar.gz", "not-tool")

    installation = _make_installation(monkeypatch, repo_path, vendor_dir)

    with pytest.raises(RuntimeError, match="Archive: sample-backend.tar.gz"):
        installation.install(InstallFromVendorPackage())
