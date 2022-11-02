# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Corstone related installation functions.."""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import pytest

from mlia.backend.manager import BackendRunner
from mlia.tools.metadata.common import DownloadAndInstall
from mlia.tools.metadata.common import InstallFromPath
from mlia.tools.metadata.corstone import BackendInfo
from mlia.tools.metadata.corstone import BackendInstallation
from mlia.tools.metadata.corstone import BackendInstaller
from mlia.tools.metadata.corstone import BackendMetadata
from mlia.tools.metadata.corstone import CompoundPathChecker
from mlia.tools.metadata.corstone import Corstone300Installer
from mlia.tools.metadata.corstone import get_corstone_300_installation
from mlia.tools.metadata.corstone import get_corstone_310_installation
from mlia.tools.metadata.corstone import get_corstone_installations
from mlia.tools.metadata.corstone import PackagePathChecker
from mlia.tools.metadata.corstone import PathChecker
from mlia.tools.metadata.corstone import StaticPathChecker


@pytest.fixture(name="test_mlia_resources")
def fixture_test_mlia_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect MLIA resources resolution to the temp directory."""
    mlia_resources = tmp_path / "resources"
    mlia_resources.mkdir()

    monkeypatch.setattr(
        "mlia.tools.metadata.corstone.get_mlia_resources",
        MagicMock(return_value=mlia_resources),
    )

    return mlia_resources


def get_backend_installation(  # pylint: disable=too-many-arguments
    backend_runner_mock: MagicMock = MagicMock(),
    name: str = "test_name",
    description: str = "test_description",
    download_artifact: MagicMock | None = None,
    path_checker: PathChecker = MagicMock(),
    apps_resources: list[str] | None = None,
    system_config: str | None = None,
    backend_installer: BackendInstaller = MagicMock(),
    supported_platforms: list[str] | None = None,
) -> BackendInstallation:
    """Get backend installation."""
    return BackendInstallation(
        backend_runner=backend_runner_mock,
        metadata=BackendMetadata(
            name=name,
            description=description,
            system_config=system_config or "",
            apps_resources=apps_resources or [],
            fvp_dir_name="sample_dir",
            download_artifact=download_artifact,
            supported_platforms=supported_platforms,
        ),
        path_checker=path_checker,
        backend_installer=backend_installer,
    )


@pytest.mark.parametrize(
    "platform, supported_platforms, expected_result",
    [
        ["Linux", ["Linux"], True],
        ["Linux", [], True],
        ["Linux", None, True],
        ["Windows", ["Linux"], False],
    ],
)
def test_could_be_installed_depends_on_platform(
    platform: str,
    supported_platforms: list[str] | None,
    expected_result: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that installation could not be installed on unsupported platform."""
    monkeypatch.setattr(
        "mlia.tools.metadata.corstone.platform.system", MagicMock(return_value=platform)
    )
    monkeypatch.setattr(
        "mlia.tools.metadata.corstone.all_paths_valid", MagicMock(return_value=True)
    )
    backend_runner_mock = MagicMock(spec=BackendRunner)

    installation = get_backend_installation(
        backend_runner_mock,
        supported_platforms=supported_platforms,
    )
    assert installation.could_be_installed == expected_result


def test_get_corstone_installations() -> None:
    """Test function get_corstone_installation."""
    installs = get_corstone_installations()
    assert len(installs) == 2
    assert all(isinstance(install, BackendInstallation) for install in installs)


def test_backend_installation_metadata_resolving() -> None:
    """Test backend installation metadata resolving."""
    backend_runner_mock = MagicMock(spec=BackendRunner)
    installation = get_backend_installation(backend_runner_mock)

    assert installation.name == "test_name"
    assert installation.description == "test_description"

    backend_runner_mock.all_installed.return_value = False
    assert installation.already_installed is False

    assert installation.could_be_installed is True


def test_backend_installation_supported_install_types(tmp_path: Path) -> None:
    """Test supported installation types."""
    installation_no_download_artifact = get_backend_installation()
    assert installation_no_download_artifact.supports(DownloadAndInstall()) is False

    installation_with_download_artifact = get_backend_installation(
        download_artifact=MagicMock()
    )
    assert installation_with_download_artifact.supports(DownloadAndInstall()) is True

    path_checker_mock = MagicMock(return_value=BackendInfo(tmp_path))
    installation_can_install_from_dir = get_backend_installation(
        path_checker=path_checker_mock
    )
    assert installation_can_install_from_dir.supports(InstallFromPath(tmp_path)) is True

    any_installation = get_backend_installation()
    assert any_installation.supports("unknown_install_type") is False  # type: ignore


def test_backend_installation_install_wrong_type() -> None:
    """Test that operation should fail if wrong install type provided."""
    with pytest.raises(Exception, match="Unable to install wrong_install_type"):
        backend_runner_mock = MagicMock(spec=BackendRunner)
        installation = get_backend_installation(backend_runner_mock)

        installation.install("wrong_install_type")  # type: ignore


def test_backend_installation_install_from_path(
    tmp_path: Path, test_mlia_resources: Path
) -> None:
    """Test installation from the path."""
    system_config = test_mlia_resources / "example_config.json"
    system_config.touch()

    sample_app = test_mlia_resources / "sample_app"
    sample_app.mkdir()

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    path_checker_mock = MagicMock(return_value=BackendInfo(dist_dir))

    backend_runner_mock = MagicMock(spec=BackendRunner)
    installation = get_backend_installation(
        backend_runner_mock=backend_runner_mock,
        path_checker=path_checker_mock,
        apps_resources=[sample_app.name],
        system_config="example_config.json",
    )

    assert installation.supports(InstallFromPath(dist_dir)) is True
    installation.install(InstallFromPath(dist_dir))

    backend_runner_mock.install_system.assert_called_once()
    backend_runner_mock.install_application.assert_called_once_with(sample_app)


@pytest.mark.parametrize("copy_source", [True, False])
def test_backend_installation_install_from_static_path(
    tmp_path: Path, test_mlia_resources: Path, copy_source: bool
) -> None:
    """Test installation from the predefined path."""
    system_config = test_mlia_resources / "example_config.json"
    system_config.touch()

    custom_system_config = test_mlia_resources / "custom_config.json"
    custom_system_config.touch()

    sample_app = test_mlia_resources / "sample_app"
    sample_app.mkdir()

    predefined_location = tmp_path / "backend"
    predefined_location.mkdir()

    predefined_location_file = predefined_location / "file.txt"
    predefined_location_file.touch()

    predefined_location_dir = predefined_location / "folder"
    predefined_location_dir.mkdir()
    nested_file = predefined_location_dir / "nested_file.txt"
    nested_file.touch()

    backend_runner_mock = MagicMock(spec=BackendRunner)

    def check_install_dir(install_dir: Path) -> None:
        """Check content of the install dir."""
        assert install_dir.is_dir()
        files = list(install_dir.iterdir())

        if copy_source:
            assert len(files) == 3
            assert all(install_dir / item in files for item in ["file.txt", "folder"])
            assert (install_dir / "folder/nested_file.txt").is_file()
        else:
            assert len(files) == 1

        assert install_dir / "custom_config.json" in files

    backend_runner_mock.install_system.side_effect = check_install_dir

    installation = get_backend_installation(
        backend_runner_mock=backend_runner_mock,
        path_checker=StaticPathChecker(
            predefined_location,
            ["file.txt"],
            copy_source=copy_source,
            system_config=str(custom_system_config),
        ),
        apps_resources=[sample_app.name],
        system_config="example_config.json",
    )

    assert installation.supports(InstallFromPath(predefined_location)) is True
    installation.install(InstallFromPath(predefined_location))

    backend_runner_mock.install_system.assert_called_once()
    backend_runner_mock.install_application.assert_called_once_with(sample_app)


def create_sample_fvp_archive(tmp_path: Path) -> Path:
    """Create sample FVP tar archive."""
    fvp_archive_dir = tmp_path / "archive"
    fvp_archive_dir.mkdir()

    sample_file = fvp_archive_dir / "sample.txt"
    sample_file.write_text("Sample file")

    sample_dir = fvp_archive_dir / "sample_dir"
    sample_dir.mkdir()

    fvp_archive = tmp_path / "archive.tgz"
    with tarfile.open(fvp_archive, "w:gz") as fvp_archive_tar:
        fvp_archive_tar.add(fvp_archive_dir, arcname=fvp_archive_dir.name)

    return fvp_archive


def test_backend_installation_download_and_install(
    test_mlia_resources: Path, tmp_path: Path
) -> None:
    """Test downloading and installation process."""
    fvp_archive = create_sample_fvp_archive(tmp_path)

    system_config = test_mlia_resources / "example_config.json"
    system_config.touch()

    download_artifact_mock = MagicMock()
    download_artifact_mock.download_to.return_value = fvp_archive

    path_checker = PackagePathChecker(["archive/sample.txt"], "archive/sample_dir")

    def installer(_eula_agreement: bool, dist_dir: Path) -> Path:
        """Sample installer."""
        return dist_dir

    backend_runner_mock = MagicMock(spec=BackendRunner)
    installation = get_backend_installation(
        backend_runner_mock,
        download_artifact=download_artifact_mock,
        backend_installer=installer,
        path_checker=path_checker,
        system_config="example_config.json",
    )

    installation.install(DownloadAndInstall())

    backend_runner_mock.install_system.assert_called_once()


@pytest.mark.parametrize(
    "dir_content, expected_result",
    [
        [
            ["models/", "file1.txt", "file2.txt"],
            "models",
        ],
        [
            ["file1.txt", "file2.txt"],
            None,
        ],
        [
            ["models/", "file2.txt"],
            None,
        ],
    ],
)
def test_corstone_path_checker_valid_path(
    tmp_path: Path, dir_content: list[str], expected_result: str | None
) -> None:
    """Test Corstone path checker valid scenario."""
    path_checker = PackagePathChecker(["file1.txt", "file2.txt"], "models")

    for item in dir_content:
        if item.endswith("/"):
            item_dir = tmp_path / item
            item_dir.mkdir()
        else:
            item_file = tmp_path / item
            item_file.touch()

    result = path_checker(tmp_path)
    expected = (
        None if expected_result is None else BackendInfo(tmp_path / expected_result)
    )

    assert result == expected


@pytest.mark.parametrize("system_config", [None, "system_config"])
@pytest.mark.parametrize("copy_source", [True, False])
def test_static_path_checker(
    tmp_path: Path, copy_source: bool, system_config: str | None
) -> None:
    """Test static path checker."""
    static_checker = StaticPathChecker(
        tmp_path, [], copy_source=copy_source, system_config=system_config
    )
    assert static_checker(tmp_path) == BackendInfo(
        tmp_path, copy_source=copy_source, system_config=system_config
    )


def test_static_path_checker_not_valid_path(tmp_path: Path) -> None:
    """Test static path checker should return None if path is not valid."""
    static_checker = StaticPathChecker(tmp_path, ["file.txt"])
    assert static_checker(tmp_path / "backend") is None


def test_static_path_checker_not_valid_structure(tmp_path: Path) -> None:
    """Test static path checker should return None if files are missing."""
    static_checker = StaticPathChecker(tmp_path, ["file.txt"])
    assert static_checker(tmp_path) is None

    missing_file = tmp_path / "file.txt"
    missing_file.touch()

    assert static_checker(tmp_path) == BackendInfo(tmp_path, copy_source=False)


def test_compound_path_checker(tmp_path: Path) -> None:
    """Test compound path checker."""
    path_checker_path_valid_path = MagicMock(return_value=BackendInfo(tmp_path))
    path_checker_path_not_valid_path = MagicMock(return_value=None)

    checker = CompoundPathChecker(
        path_checker_path_valid_path, path_checker_path_not_valid_path
    )
    assert checker(tmp_path) == BackendInfo(tmp_path)

    checker = CompoundPathChecker(path_checker_path_not_valid_path)
    assert checker(tmp_path) is None


@pytest.mark.parametrize(
    "eula_agreement, expected_command",
    [
        [
            True,
            [
                "./FVP_Corstone_SSE-300.sh",
                "-q",
                "-d",
                "corstone-300",
            ],
        ],
        [
            False,
            [
                "./FVP_Corstone_SSE-300.sh",
                "-q",
                "-d",
                "corstone-300",
                "--nointeractive",
                "--i-agree-to-the-contained-eula",
            ],
        ],
    ],
)
def test_corstone_300_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eula_agreement: bool,
    expected_command: list[str],
) -> None:
    """Test Corstone-300 installer."""
    command_mock = MagicMock()

    monkeypatch.setattr(
        "mlia.tools.metadata.corstone.subprocess.check_call", command_mock
    )
    installer = Corstone300Installer()
    result = installer(eula_agreement, tmp_path)

    command_mock.assert_called_once_with(expected_command)
    assert result == tmp_path / "corstone-300"


@pytest.mark.parametrize(
    "corstone_installation, expected_paths",
    [
        [
            get_corstone_300_installation(),
            {
                "/opt/VHT/VHT_Corstone_SSE-300_Ethos-U55",
                "/opt/VHT/VHT_Corstone_SSE-300_Ethos-U65",
            },
        ],
        [
            get_corstone_310_installation(),
            {
                "/opt/VHT/VHT_Corstone_SSE-310",
                "/opt/VHT/VHT_Corstone_SSE-310_Ethos-U65",
            },
        ],
    ],
)
def test_corstone_vht_install(
    corstone_installation: BackendInstallation,
    expected_paths: set,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test if Corstone 300/310 could be installed from /opt/VHT."""

    def _all_files_exist(paths: Iterable[Path]) -> bool:
        """Check if all files exist."""
        pathset = {item.as_posix() for item in paths}
        return pathset == expected_paths

    create_destination_and_install_mock = MagicMock()

    monkeypatch.setattr(
        "mlia.tools.metadata.corstone.all_files_exist", _all_files_exist
    )

    monkeypatch.setattr("mlia.backend.system.get_available_systems", lambda: [])

    monkeypatch.setattr(
        "mlia.backend.system.create_destination_and_install",
        create_destination_and_install_mock,
    )

    corstone_installation.install(InstallFromPath(Path("/opt/VHT")))

    create_destination_and_install_mock.assert_called_once()


def test_corstone_uninstall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the uninstall function in Corstone."""
    remove_system_mock = MagicMock()

    monkeypatch.setattr(
        "mlia.tools.metadata.corstone.remove_system",
        remove_system_mock,
    )

    installation = get_corstone_300_installation()

    installation.uninstall()
    remove_system_mock.assert_called_once_with("corstone_300")
