# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend installation from vendored archives."""

from __future__ import annotations

import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
from unittest.mock import MagicMock

import pytest

from mlia.backend.install import (
    BackendInstallation,
    CompoundPathChecker,
    DownloadAndInstall,
    EulaDocument,
    InstallFromVendorPackage,
    PackagePathChecker,
    PyPackageBackendInstallation,
    _filter_tar_members,
    logger,
)
from mlia.backend.repo import get_backend_repository
from mlia.utils.download import DownloadConfig


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


def test_archive_filter_accepts_destination_with_symlinked_parent(
    tmp_path: Path,
) -> None:
    """Resolved destination aliases should not reject safe archive members."""
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    destination = alias_parent / "dist"
    destination.mkdir()
    member = tarfile.TarInfo("tool")

    assert list(_filter_tar_members([member], destination)) == [member]


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


@pytest.mark.parametrize("eula_agreement", [False, True])
def test_install_from_vendor_archive_passes_eula_agreement_to_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
    eula_agreement: bool,
) -> None:
    """Pass the caller's EULA decision to vendored backend installers."""
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    archive_path = vendor_dir / "sample-backend.tar.gz"
    _create_tar_with_file(archive_path, "payload")
    backend_installer = MagicMock()

    def install_backend(agreement: bool, dist_dir: Path) -> Path:
        backend_installer(agreement, dist_dir)
        installed_dir = dist_dir / "installed"
        installed_dir.mkdir()
        (installed_dir / "tool").write_text("data", encoding="utf-8")
        return installed_dir

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
        backend_installer=install_backend,
        vendor_path="sample-backend",
    )

    installation.install(InstallFromVendorPackage(eula_agreement=eula_agreement))

    backend_installer.assert_called_once()
    assert backend_installer.call_args.args[0] is eula_agreement
    assert (fake_backend_repository_path / "backends/sample-backend/tool").is_file()


def test_install_from_vendor_archive_prompts_with_contained_eula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Display contained terms and reuse the extracted artifact after consent."""
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    source_dir = tmp_path / "archive-source"
    (source_dir / "license_terms").mkdir(parents=True)
    (source_dir / "license_terms" / "terms.txt").write_text(
        "These are the licence terms.\n", encoding="utf-8"
    )
    (source_dir / "payload").write_text("data", encoding="utf-8")
    _create_tar_from_directory(vendor_dir / "sample-backend.tar.gz", source_dir)
    backend_installer = MagicMock()

    def install_backend(agreement: bool, dist_dir: Path) -> Path:
        backend_installer(agreement, dist_dir)
        installed_dir = dist_dir / "installed"
        installed_dir.mkdir()
        (installed_dir / "tool").write_text("data", encoding="utf-8")
        return installed_dir

    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )
    prompt = MagicMock(return_value="yes")
    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(expected_files=["tool"]),
        backend_installer=install_backend,
        vendor_path="sample-backend",
        requires_eula=True,
    )

    installation.install(
        InstallFromVendorPackage(eula_agreement=False, eula_prompt=prompt)
    )

    message = prompt.call_args.args[0]
    assert "End-user licence agreement for backend 'sample-backend'" in message
    assert "terms.txt" in message
    assert "These are the licence terms." in message
    assert message.endswith("Accept these terms and install the backend?")
    backend_installer.assert_called_once()
    assert backend_installer.call_args.args[0] is True
    assert (fake_backend_repository_path / "backends/sample-backend/tool").is_file()


def test_download_prompts_from_and_installs_the_same_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Download once, show its contained EULA, and continue from its staging area."""
    source_dir = tmp_path / "archive-source"
    (source_dir / "license_terms").mkdir(parents=True)
    (source_dir / "license_terms" / "terms.txt").write_text(
        "Downloaded licence terms.\n", encoding="utf-8"
    )
    archive = tmp_path / "source" / "sample-backend.tar.gz"
    _create_tar_from_directory(archive, source_dir)
    download = MagicMock(side_effect=lambda dest, **_: copyfile(archive, dest))
    monkeypatch.setattr("mlia.backend.install.download", download)

    def install_backend(agreement: bool, dist_dir: Path) -> Path:
        assert agreement is True
        assert (dist_dir / "license_terms/terms.txt").is_file()
        installed_dir = dist_dir / "installed"
        installed_dir.mkdir()
        (installed_dir / "tool").write_text("data", encoding="utf-8")
        return installed_dir

    prompt = MagicMock(return_value="yes")
    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=DownloadConfig(
            "https://example.com/sample-backend.tar.gz", sha256_hash="unused"
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(expected_files=["tool"]),
        backend_installer=install_backend,
        requires_eula=True,
    )

    installation.install(DownloadAndInstall(eula_agreement=False, eula_prompt=prompt))

    download.assert_called_once()
    assert "Downloaded licence terms." in prompt.call_args.args[0]
    assert (fake_backend_repository_path / "backends/sample-backend/tool").is_file()


@pytest.mark.parametrize("response", ["no", ""])
def test_install_from_vendor_archive_declining_eula_stops_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
    response: str,
) -> None:
    """Do not invoke the backend installer when contained terms are declined."""
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    source_dir = tmp_path / "archive-source"
    source_dir.mkdir()
    (source_dir / "terms").write_text("terms", encoding="utf-8")
    _create_tar_from_directory(vendor_dir / "sample-backend.tar.gz", source_dir)
    backend_installer = MagicMock()
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
        backend_installer=backend_installer,
        vendor_path="sample-backend",
        requires_eula=True,
        eula_resolver=lambda dist_dir: (
            [EulaDocument("terms", (dist_dir / "terms").read_text())]
            if dist_dir is not None
            else []
        ),
    )

    with pytest.raises(RuntimeError, match="EULA.*was not accepted"):
        installation.install(
            InstallFromVendorPackage(
                eula_agreement=False, eula_prompt=MagicMock(return_value=response)
            )
        )

    backend_installer.assert_not_called()


def test_install_from_vendor_archive_blocks_when_eula_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend_repository_path: Path,
) -> None:
    """Never offer blind acceptance when required terms cannot be resolved."""
    vendor_dir = tmp_path / "vendor" / "sample-backend"
    _create_tar_with_file(vendor_dir / "sample-backend.tar.gz", "payload")
    backend_installer = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.install.vendor_artifact_path", lambda _: vendor_dir
    )
    prompt = MagicMock(return_value="yes")
    installation = BackendInstallation(
        name="sample-backend",
        description="Sample backend",
        fvp_dir_name="sample-backend",
        download_config=None,
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(expected_files=["tool"]),
        backend_installer=backend_installer,
        vendor_path="sample-backend",
        requires_eula=True,
    )

    with pytest.raises(RuntimeError, match="licence terms could not be found"):
        installation.install(
            InstallFromVendorPackage(eula_agreement=False, eula_prompt=prompt)
        )

    prompt.assert_not_called()
    backend_installer.assert_not_called()


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


@pytest.mark.parametrize(
    ("agreement", "response", "should_install"),
    [
        pytest.param(True, None, True, id="pre-approved"),
        pytest.param(False, "yes", True, id="prompt-approved"),
        pytest.param(False, "no", False, id="prompt-declined"),
    ],
)
def test_python_package_installation_resolves_eula(
    monkeypatch: pytest.MonkeyPatch,
    agreement: bool,
    response: str | None,
    should_install: bool,
) -> None:
    """Python package installations can expose their EULA to frontends."""
    package_manager = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.install.get_package_manager", lambda: package_manager
    )
    resolver = MagicMock(
        return_value=[EulaDocument("Licence", "Terms for the package")]
    )
    prompt = MagicMock(return_value=response) if response is not None else MagicMock()
    installation = PyPackageBackendInstallation(
        description="Sample backend",
        name="sample-backend",
        packages_to_install=["sample-package"],
        packages_to_uninstall=["sample-package"],
        expected_packages=["sample-package"],
        requires_eula=True,
        eula_resolver=resolver,
    )

    install_type = DownloadAndInstall(
        eula_agreement=agreement,
        eula_prompt=prompt,
    )
    if should_install:
        resolved_install_type = installation.resolve_eula_prompt(install_type)
        package_manager.install.assert_not_called()
        installation.install(resolved_install_type)
        package_manager.install.assert_called_once_with(["sample-package"])
    else:
        with pytest.raises(RuntimeError, match="was not accepted"):
            installation.resolve_eula_prompt(install_type)
        package_manager.install.assert_not_called()

    if agreement:
        resolver.assert_not_called()
        prompt.assert_not_called()
    else:
        resolver.assert_called_once_with(None)
        prompt.assert_called_once()


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
