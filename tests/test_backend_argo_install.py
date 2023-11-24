# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for docker installation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.argo.install import ARGO_PATH_ENV_VAR
from mlia.backend.argo.install import DOCKER_IMAGE_NAME
from mlia.backend.argo.install import DockerInstallation
from mlia.backend.argo.install import get_argo_installation
from mlia.backend.install import ARTIFACTORY_PASSWORD_ENV_VAR
from mlia.backend.install import ARTIFACTORY_USERNAME_ENV_VAR
from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import InstallFromPath
from mlia.utils.misc import is_docker_available


@pytest.fixture(name="docker_installation")
def fixture_docker_installation() -> Any:
    """Fixture to create a DockerInstallation instance with test values."""
    yield get_argo_installation()


def mock_docker(
    monkeypatch: pytest.MonkeyPatch,
    docker_client_mock: Any,
) -> None:
    """Mock the docker module with the given client mock instance."""
    docker_mock = MagicMock()
    docker_mock.from_env = MagicMock(return_value=docker_client_mock)
    monkeypatch.setattr("mlia.backend.argo.install.docker", docker_mock)


@pytest.mark.skipif(
    condition=not is_docker_available(), reason="Docker is not available."
)
def test_docker_installation_hello_world() -> None:
    """Test installation of 'hello-world' from docker hub."""
    install = DockerInstallation(
        "hello-world",
        "Test using 'hello-world' from the official docker hub.",
        "hello-world",
        registry=None,
    )
    assert install.could_be_installed
    if install.already_installed:
        install.uninstall()
    assert not install.already_installed
    install.install(DownloadAndInstall())
    assert install.already_installed


@pytest.mark.parametrize(
    ("kwargs", "repository"),
    (
        (
            {
                "name": "test",
                "description": "description",
                "image_name": "image",
                "registry": None,
            },
            "image",
        ),
        (
            {
                "name": "test",
                "description": "description",
                "image_name": "image",
                "registry": "ml-tooling--docker-local.artifactory.geo.arm.com",
            },
            "ml-tooling--docker-local.artifactory.geo.arm.com/image",
        ),
    ),
)
def test_docker_installation_init(kwargs: dict[str, Any], repository: str) -> None:
    """Test initialization of class 'DockerInstallation'."""
    install = DockerInstallation(**kwargs)
    assert all(getattr(install, name) == value for name, value in kwargs.items())
    assert install.repository == repository


def test_docker_installation_supports(docker_installation: DockerInstallation) -> None:
    """Test function supports() of class 'DockerInstallation'."""
    assert docker_installation.supports(DownloadAndInstall())
    assert not docker_installation.supports(InstallFromPath(Path("DOES_NOT_EXIST")))


@pytest.mark.parametrize("docker_available", (True, False))
def test_docker_installation_could_be_installed(
    monkeypatch: pytest.MonkeyPatch,
    docker_installation: DockerInstallation,
    docker_available: bool,
) -> None:
    """Test function could_be_installed() of class 'DockerInstallation'."""
    monkeypatch.setattr(
        "mlia.backend.argo.install.is_docker_available_cached",
        MagicMock(return_value=docker_available),
    )
    assert docker_installation.could_be_installed == docker_available


def test_docker_installation_install(
    monkeypatch: pytest.MonkeyPatch, docker_installation: DockerInstallation
) -> None:
    """Test DockerInstallation.install()."""
    with pytest.raises(TypeError):
        docker_installation.install(InstallFromPath(Path("DOES_NOT_EXIST")))

    docker_client_mock = MagicMock()
    docker_client_mock.api.pull = MagicMock(
        return_value=(
            {"id": 123, "status": "test1"},
            {
                "id": 123,
                "status": "test2",
                "progressDetail": {},
            },
            {
                "id": 123,
                "status": "test3",
                "progressDetail": {"current": 1, "total": 2},
            },
        )
    )
    docker_tag_mock = MagicMock()
    docker_client_mock.api.tag = docker_tag_mock
    mock_docker(monkeypatch, docker_client_mock)

    with pytest.raises(RuntimeError):
        # Credentials are not set as env vars => raises an exception
        docker_installation.install(DownloadAndInstall())

    monkeypatch.setenv(ARTIFACTORY_USERNAME_ENV_VAR, "abc123@arm.com")
    monkeypatch.setenv(ARTIFACTORY_PASSWORD_ENV_VAR, "1234")

    docker_installation.install(DownloadAndInstall())

    docker_tag_mock.return_value = False
    with pytest.raises(RuntimeError):
        docker_installation.install(DownloadAndInstall())


def test_docker_installation_uninstall(
    monkeypatch: pytest.MonkeyPatch, docker_installation: DockerInstallation
) -> None:
    """Test DockerInstallation.uninstall()."""
    docker_client_mock = MagicMock()
    docker_client_mock.images.list = MagicMock(
        return_value=(MagicMock(), MagicMock()),
    )
    remove_mock = MagicMock()
    docker_client_mock.images.remove = remove_mock
    mock_docker(monkeypatch, docker_client_mock)

    docker_installation.uninstall()
    assert remove_mock.call_count == 2


@pytest.mark.parametrize(
    ("images", "expected_result"),
    (([], False), ([MagicMock()], True), ([MagicMock(), MagicMock()], True)),
)
def test_docker_installation_already_installed(
    monkeypatch: pytest.MonkeyPatch,
    docker_installation: DockerInstallation,
    images: list,
    expected_result: bool,
) -> None:
    """Test DockerInstallation.uninstall()."""
    docker_client_mock = MagicMock()
    docker_client_mock.images.list = MagicMock(
        return_value=images,
    )
    mock_docker(monkeypatch, docker_client_mock)

    assert docker_installation.already_installed == expected_result


def test_get_argo_installation() -> None:
    """Test function get_argo_installation()."""
    installation = get_argo_installation()
    assert installation.name == "argo"
    assert installation.image_name == DOCKER_IMAGE_NAME


@pytest.mark.parametrize(
    ("env_var_name", "env_var_value", "exec_path"),
    (
        ("ENV_VAR", "some-file", Path("some-file")),
        ("ENV_VAR", None, None),
        (None, None, None),
    ),
)
def test_executable_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    env_var_name: str | None,
    env_var_value: str | None,
    exec_path: Path | None,
) -> None:
    """Test property 'DockerInstallation.executable_overwrite'."""
    if env_var_name:
        monkeypatch.setenv(env_var_name, env_var_value if env_var_value else "")
    installation = DockerInstallation(
        "Name", "Description", "IMAGE", "REGISTRY", env_var_name
    )
    assert installation.executable_overwrite == exec_path


def test_executable_overwrite_argo(docker_installation: DockerInstallation) -> None:
    """Make sure the right environment variable is used for Argo."""
    assert (
        # pylint: disable=protected-access
        docker_installation._executable_overwrite_env_var
        == ARGO_PATH_ENV_VAR
    )
