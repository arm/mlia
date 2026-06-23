# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend installation from vendored archives."""

from __future__ import annotations

import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.install import (
    BackendInstallation,
    CompoundPathChecker,
    InstallFromVendorPackage,
    PackagePathChecker,
    logger,
)
from mlia.backend.repo import get_backend_repository


@pytest.fixture
def fake_backend_repository_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    repo_path = tmp_path / "repo"
    repo = get_backend_repository(repo_path)
    monkeypatch.setattr("mlia.backend.install.get_backend_repository", lambda: repo)
    monkeypatch.setattr(
        "mlia.backend.install.temp_directory",
        lambda: _temporary_directory(tmp_path / "extract"),
    )
    return repo_path


def _create_tar_with_file(archive_path: Path, filename: str) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    file_path = archive_path.parent / filename
    file_path.write_text("data", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(file_path, arcname=filename)
    file_path.unlink()


def _create_tar_from_directory(archive_path: Path, source_dir: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(source_dir))


@contextmanager
def _temporary_directory(path: Path) -> Iterator[Path]:
    path.mkdir(parents=True, exist_ok=True)
    yield path


def test_install_from_vendor_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Install backend when vendored artifacts contain a tar.gz archive."""
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

    backend_dir = fake_backend_repository_path / "backends" / "sample-backend"
    assert backend_dir.is_dir()
    assert (backend_dir / "tool").is_file()


def test_install_from_vendor_archive_backend_subfolder_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Install backend files fails if backend subfolder does not exist."""
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    source_dir = tmp_path / "archive-source"
    source_dir.mkdir()
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(manifest_path, arcname="manifest.json")

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )
    logger_debug = MagicMock()
    monkeypatch.setattr(logger, "debug", logger_debug)

    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=["manifest.json"],
            backend_subfolder="backend",
        ),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )

    with pytest.raises(
        RuntimeError, match="Unable to resolve backend path from vendor archive"
    ):
        installation.install(InstallFromVendorPackage())
    logger_debug.assert_called_with("Backend subfolder 'backend' not found in archive.")


def test_install_from_vendor_archive_backend_subfolder_is_not_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Install backend files fails if backend subfolder is not a folder."""

    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    source_dir = tmp_path / "archive-source"
    source_dir.mkdir()
    (source_dir / "backend").write_text("not a directory", encoding="utf-8")
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(manifest_path, arcname="manifest.json")
        archive.add(source_dir / "backend", arcname="backend")

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )
    logger_debug = MagicMock()
    monkeypatch.setattr(logger, "debug", logger_debug)

    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=["manifest.json"],
            backend_subfolder="backend",
        ),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )

    with pytest.raises(
        RuntimeError, match="Unable to resolve backend path from vendor archive"
    ):
        installation.install(InstallFromVendorPackage())
    logger_debug.assert_called_with("Backend subfolder 'backend' not found in archive.")


def test_install_from_vendor_archive_uses_backend_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Install backend files from the configured package subfolder."""
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    source_dir = tmp_path / "archive-source"
    backend_dir = source_dir / "backend"
    backend_dir.mkdir(parents=True)
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    tool_path = backend_dir / "tool"
    tool_path.write_text("data", encoding="utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(manifest_path, arcname="manifest.json")
        archive.add(tool_path, arcname="backend/tool")

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )

    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=["manifest.json"],
            backend_subfolder="backend",
        ),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )

    installation.install(InstallFromVendorPackage())

    installed_backend_dir = fake_backend_repository_path / "backends" / "sample-backend"
    assert (installed_backend_dir / "tool").is_file()
    assert not (installed_backend_dir / "manifest.json").exists()


def test_install_from_vendor_archive_copies_supporting_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Install primary backend files with an additional supporting folder."""
    source_dir = tmp_path / "archive-source"
    primary_dir = source_dir / "primary"
    supporting_dir = source_dir / "support"
    primary_dir.mkdir(parents=True)
    supporting_dir.mkdir()
    (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (primary_dir / "tool").write_text("data", encoding="utf-8")
    (supporting_dir / "runtime.so").write_text("support", encoding="utf-8")

    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    _create_tar_from_directory(archive_path, source_dir)

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )

    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=["manifest.json", "primary/tool", "support/runtime.so"],
            backend_subfolder="primary",
            supporting_subfolders=["support"],
        ),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )

    installation.install(InstallFromVendorPackage())

    installed_backend_dir = fake_backend_repository_path / "backends" / "sample-backend"
    assert (installed_backend_dir / "tool").is_file()
    assert (installed_backend_dir / "support" / "runtime.so").is_file()
    assert not (installed_backend_dir / "manifest.json").exists()


def test_install_from_vendor_archive_copies_nested_supporting_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Install a supporting folder with nested contents."""
    source_dir = tmp_path / "archive-source"
    primary_dir = source_dir / "primary"
    nested_support_dir = source_dir / "support" / "lib" / "site-packages" / "pkg"
    primary_dir.mkdir(parents=True)
    nested_support_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (primary_dir / "tool").write_text("data", encoding="utf-8")
    (nested_support_dir / "__init__.py").write_text("", encoding="utf-8")

    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    _create_tar_from_directory(archive_path, source_dir)

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )

    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=["manifest.json", "primary/tool"],
            backend_subfolder="primary",
            supporting_subfolders=["support"],
        ),
        backend_installer=None,
        dependencies=[],
        vendor_path="sample-backend",
    )

    installation.install(InstallFromVendorPackage())

    installed_backend_dir = fake_backend_repository_path / "backends" / "sample-backend"
    assert (installed_backend_dir / "tool").is_file()
    assert (
        installed_backend_dir
        / "support"
        / "lib"
        / "site-packages"
        / "pkg"
        / "__init__.py"
    ).is_file()


def test_package_path_checker_resolves_supporting_subfolder(tmp_path: Path) -> None:
    """Resolve supporting folders with the primary backend subfolder."""
    package_dir = tmp_path / "package"
    primary_dir = package_dir / "primary"
    supporting_dir = package_dir / "support"
    primary_dir.mkdir(parents=True)
    supporting_dir.mkdir()
    (package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (supporting_dir / "runtime.so").write_text("support", encoding="utf-8")

    backend_info = PackagePathChecker(
        expected_files=["manifest.json"],
        backend_subfolder="primary",
        supporting_subfolders=["support"],
    )(package_dir)

    assert backend_info is not None
    assert backend_info.backend_path == primary_dir
    assert backend_info.supporting_paths == [(supporting_dir, Path("support"))]


def test_package_path_checker_rejects_missing_supporting_subfolder(
    tmp_path: Path,
) -> None:
    """Reject package layouts missing a configured supporting folder."""
    package_dir = tmp_path / "package"
    primary_dir = package_dir / "primary"
    primary_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (primary_dir / "tool").write_text("data", encoding="utf-8")

    backend_info = PackagePathChecker(
        expected_files=["manifest.json"],
        backend_subfolder="primary",
        supporting_subfolders=["support"],
    )(package_dir)

    assert backend_info is None


def test_package_path_checker_rejects_file_supporting_subfolder(
    tmp_path: Path,
) -> None:
    """Reject package layouts where a supporting folder is a file."""
    package_dir = tmp_path / "package"
    primary_dir = package_dir / "primary"
    primary_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (primary_dir / "tool").write_text("data", encoding="utf-8")
    (package_dir / "support").write_text("not a directory", encoding="utf-8")

    backend_info = PackagePathChecker(
        expected_files=["manifest.json"],
        backend_subfolder="primary",
        supporting_subfolders=["support"],
    )(package_dir)

    assert backend_info is None


def test_package_path_checker_rejects_outside_supporting_subfolder(
    tmp_path: Path,
) -> None:
    """Reject supporting folders outside the package root."""
    package_dir = tmp_path / "package"
    primary_dir = package_dir / "primary"
    outside_dir = tmp_path / "outside"
    primary_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (primary_dir / "tool").write_text("data", encoding="utf-8")
    (outside_dir / "runtime.so").write_text("support", encoding="utf-8")

    backend_info = PackagePathChecker(
        expected_files=["manifest.json"],
        backend_subfolder="primary",
        supporting_subfolders=["../outside"],
    )(package_dir)

    assert backend_info is None


def test_compound_path_checker_keeps_first_matching_layout(
    tmp_path: Path,
) -> None:
    """Return all payloads from the first matching checker only."""
    package_dir = tmp_path / "package"
    first_primary_dir = package_dir / "first" / "primary"
    first_supporting_dir = package_dir / "first" / "support"
    second_primary_dir = package_dir / "second" / "primary"
    first_primary_dir.mkdir(parents=True)
    first_supporting_dir.mkdir()
    second_primary_dir.mkdir(parents=True)
    (first_primary_dir / "tool").write_text("data", encoding="utf-8")
    (first_supporting_dir / "runtime.so").write_text("support", encoding="utf-8")
    (second_primary_dir / "tool").write_text("data", encoding="utf-8")

    backend_info = CompoundPathChecker(
        PackagePathChecker(
            expected_files=["first/primary/tool"],
            backend_subfolder="first/primary",
            supporting_subfolders=["first/support"],
        ),
        PackagePathChecker(
            expected_files=["second/primary/tool"],
            backend_subfolder="second/primary",
        ),
    )(package_dir)

    assert backend_info is not None
    assert backend_info.backend_path == first_primary_dir
    assert backend_info.supporting_paths == [
        (first_supporting_dir, Path("first/support"))
    ]


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
