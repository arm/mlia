# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for installation manager."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import Installation
from mlia.backend.install import InstallationType
from mlia.backend.install import InstallFromPath
from mlia.backend.manager import DefaultInstallationManager
from mlia.core.errors import ConfigurationError
from mlia.core.errors import InternalError


def get_default_installation_manager_mock(
    name: str,
    already_installed: bool = False,
    dependencies: list[str] | None = None,
) -> MagicMock:
    """Get mock instance for DefaultInstallationManager."""
    mock = MagicMock(spec=DefaultInstallationManager)

    props = {
        "name": name,
        "already_installed": already_installed,
        "dependencies": dependencies if dependencies else [],
    }
    for prop, value in props.items():
        setattr(type(mock), prop, PropertyMock(return_value=value))

    return mock


def _ready_for_uninstall_mock() -> MagicMock:
    return get_default_installation_manager_mock(
        name="already_installed",
        already_installed=True,
    )


def get_installation_mock(
    name: str,
    already_installed: bool = False,
    could_be_installed: bool = False,
    supported_install_type: type | tuple | None = None,
    dependencies: list[str] | None = None,
) -> MagicMock:
    """Get mock instance for the installation."""
    mock = MagicMock(spec=Installation)

    def supports(install_type: InstallationType) -> bool:
        if supported_install_type is None:
            return False

        return isinstance(install_type, supported_install_type)

    mock.supports.side_effect = supports

    props = {
        "name": name,
        "already_installed": already_installed,
        "could_be_installed": could_be_installed,
        "dependencies": dependencies if dependencies else [],
    }
    for prop, value in props.items():
        setattr(type(mock), prop, PropertyMock(return_value=value))

    return mock


_already_installed_mock = partial(
    get_installation_mock,
    name="already_installed",
    already_installed=True,
    supported_install_type=(DownloadAndInstall, InstallFromPath),
)


_ready_for_installation_mock = partial(
    get_installation_mock,
    name="ready_for_installation",
    already_installed=False,
    could_be_installed=True,
)


_could_be_downloaded_and_installed_mock = partial(
    get_installation_mock,
    name="could_be_downloaded_and_installed",
    already_installed=False,
    could_be_installed=True,
    supported_install_type=DownloadAndInstall,
)


_could_be_installed_from_mock = partial(
    get_installation_mock,
    name="could_be_installed_from",
    already_installed=False,
    could_be_installed=True,
    supported_install_type=InstallFromPath,
)

_already_installed_dep_mock = partial(
    get_installation_mock,
    name="already_installed_dep",
    already_installed=True,
    supported_install_type=(DownloadAndInstall, InstallFromPath),
)


def get_installation_manager(
    noninteractive: bool,
    installations: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    yes_response: bool = True,
) -> DefaultInstallationManager:
    """Get installation manager instance."""
    if not noninteractive:
        return get_interactive_installation_manager(
            installations, monkeypatch, MagicMock(return_value=yes_response)
        )

    return DefaultInstallationManager(installations, noninteractive=noninteractive)


def get_interactive_installation_manager(
    installations: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    mock_interaction: MagicMock,
) -> DefaultInstallationManager:
    """Get and interactive installation manager instance using the given mock."""
    monkeypatch.setattr("mlia.backend.manager.yes", mock_interaction)
    return DefaultInstallationManager(installations, noninteractive=False)


def test_installation_manager_filtering() -> None:
    """Test default installation manager."""
    already_installed = _already_installed_mock()
    ready_for_installation = _ready_for_installation_mock()
    could_be_downloaded_and_installed = _could_be_downloaded_and_installed_mock()

    manager = DefaultInstallationManager(
        [
            already_installed,
            ready_for_installation,
            could_be_downloaded_and_installed,
        ]
    )
    assert manager.already_installed("already_installed") == [already_installed]
    assert manager.ready_for_installation() == [
        ready_for_installation,
        could_be_downloaded_and_installed,
    ]


@pytest.mark.parametrize("noninteractive", [True, False])
@pytest.mark.parametrize(
    "install_mock, eula_agreement, backend_name, force, expected_call",
    [
        [
            _could_be_downloaded_and_installed_mock(),
            True,
            "could_be_downloaded_and_installed",
            False,
            [call(DownloadAndInstall(eula_agreement=True))],
        ],
        [
            _could_be_downloaded_and_installed_mock(),
            False,
            "could_be_downloaded_and_installed",
            True,
            [call(DownloadAndInstall(eula_agreement=False))],
        ],
        [
            _already_installed_mock(),
            False,
            "already_installed",
            True,
            [call(DownloadAndInstall(eula_agreement=False))],
        ],
        [
            _could_be_downloaded_and_installed_mock(),
            False,
            "unknown",
            True,
            [],
        ],
    ],
)
def test_installation_manager_download_and_install(
    install_mock: MagicMock,
    noninteractive: bool,
    eula_agreement: bool,
    backend_name: str,
    force: bool,
    expected_call: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test installation process."""
    install_mock.reset_mock()

    manager = get_installation_manager(noninteractive, [install_mock], monkeypatch)

    manager.download_and_install(
        backend_name, eula_agreement=eula_agreement, force=force
    )

    assert install_mock.install.mock_calls == expected_call
    if force and install_mock.already_installed:
        install_mock.uninstall.assert_called_once()
    else:
        install_mock.uninstall.assert_not_called()


@pytest.mark.parametrize("noninteractive", [True, False])
@pytest.mark.parametrize(
    "install_mock, backend_name, force, expected_call",
    [
        [
            _could_be_installed_from_mock(),
            "could_be_installed_from",
            False,
            [call(InstallFromPath(Path("some_path")))],
        ],
        [
            _could_be_installed_from_mock(),
            "unknown",
            False,
            [],
        ],
        [
            _could_be_installed_from_mock(),
            "unknown",
            True,
            [],
        ],
        [
            _already_installed_mock(),
            "already_installed",
            False,
            [],
        ],
        [
            _already_installed_mock(),
            "already_installed",
            True,
            [call(InstallFromPath(Path("some_path")))],
        ],
    ],
)
def test_installation_manager_install_from(
    install_mock: MagicMock,
    noninteractive: bool,
    backend_name: str,
    force: bool,
    expected_call: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test installation process."""
    install_mock.reset_mock()

    manager = get_installation_manager(noninteractive, [install_mock], monkeypatch)
    manager.install_from(Path("some_path"), backend_name, force=force)

    assert install_mock.install.mock_calls == expected_call
    if force and install_mock.already_installed:
        install_mock.uninstall.assert_called_once()
    else:
        install_mock.uninstall.assert_not_called()


def test_installation_manager_unsupported_install_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that installation could not be installed via unsupported type."""
    download_install_mock = _could_be_downloaded_and_installed_mock()
    install_from_mock = _could_be_installed_from_mock()
    install_mocks = [download_install_mock, install_from_mock]

    manager = get_installation_manager(False, install_mocks, monkeypatch)
    manager.install_from(tmp_path, "could_be_downloaded_and_installed")

    manager.download_and_install("could_be_installed_from")

    for mock in install_mocks:
        mock.install.assert_not_called()
        mock.uninstall.assert_not_called()


@pytest.mark.parametrize("noninteractive", [True, False])
@pytest.mark.parametrize(
    "install_mock, backend_name, expected_call",
    [
        [
            _ready_for_uninstall_mock(),
            "already_installed",
            [call()],
        ],
    ],
)
def test_installation_manager_uninstall(
    install_mock: MagicMock,
    noninteractive: bool,
    backend_name: str,
    expected_call: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test uninstallation."""
    install_mock.reset_mock()

    manager = get_installation_manager(noninteractive, [install_mock], monkeypatch)
    manager.uninstall(backend_name)

    assert install_mock.uninstall.mock_calls == expected_call


def test_installation_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that manager should be able to detect wrong state."""
    install_mock = _ready_for_uninstall_mock()
    manager = get_installation_manager(False, [install_mock, install_mock], monkeypatch)

    with pytest.raises(
        InternalError,
        match="More than one installed backend with name already_installed found",
    ):
        manager.uninstall("already_installed")


def test_uninstall_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that uninstall should fail for uknown backend."""
    install_mock = _ready_for_uninstall_mock()
    manager = get_installation_manager(False, [install_mock, install_mock], monkeypatch)

    with pytest.raises(
        ConfigurationError, match="Backend 'some_backend' is not installed"
    ):
        manager.uninstall("some_backend")


def test_show_env_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test method show_env_details."""
    ready_to_install_mock = _ready_for_installation_mock()
    could_be_installed_mock = _could_be_installed_from_mock()

    manager = get_installation_manager(
        False,
        [ready_to_install_mock, could_be_installed_mock],
        monkeypatch,
    )
    manager.show_env_details()


@pytest.mark.parametrize(
    "dependency",
    (
        _ready_for_installation_mock(),
        _already_installed_mock(),
    ),
)
@pytest.mark.parametrize("yes_response", (True, False))
def test_could_be_installed_with_dep(
    dependency: MagicMock,
    yes_response: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test installation with a dependency."""
    install_mock = _could_be_installed_from_mock(dependencies=[dependency.name])

    yes_mock = MagicMock(return_value=yes_response)
    manager = get_interactive_installation_manager(
        [install_mock, dependency], monkeypatch, yes_mock
    )
    manager.install_from(tmp_path, install_mock.name)

    if yes_response:
        install_mock.install.assert_called_once()
    else:
        install_mock.install.assert_not_called()
    install_mock.uninstall.assert_not_called()

    dependency.install.assert_not_called()
    dependency.uninstall.assert_not_called()


def test_install_with_unknown_dep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test installation with an unknown dependency."""
    install_mock = _could_be_installed_from_mock(dependencies=["UNKNOWN_BACKEND"])

    manager = get_installation_manager(False, [install_mock], monkeypatch)
    with pytest.raises(ValueError):
        manager.install_from(tmp_path, install_mock.name)

    install_mock.install.assert_not_called()
    install_mock.uninstall.assert_not_called()


@pytest.mark.parametrize("yes_response", (True, False))
def test_uninstall_with_dep(
    yes_response: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test uninstalling a backend with a dependency."""
    dependency = _already_installed_dep_mock()
    install_mock = _already_installed_mock(dependencies=[dependency.name])
    yes_mock = MagicMock(return_value=yes_response)
    manager = get_interactive_installation_manager(
        [install_mock, dependency], monkeypatch, yes_mock
    )
    manager.uninstall(install_mock.name)

    install_mock.install.assert_not_called()
    if yes_response:
        install_mock.uninstall.assert_called_once()
    else:
        install_mock.uninstall.assert_not_called()

    dependency.install.assert_not_called()
    dependency.uninstall.assert_not_called()
