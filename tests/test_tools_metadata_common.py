# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for commmon installation related functions."""
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

from mlia.tools.metadata.common import DefaultInstallationManager
from mlia.tools.metadata.common import DownloadAndInstall
from mlia.tools.metadata.common import Installation
from mlia.tools.metadata.common import InstallationType
from mlia.tools.metadata.common import InstallFromPath


def get_installation_mock(
    name: str,
    already_installed: bool = False,
    could_be_installed: bool = False,
    supported_install_type: Optional[type] = None,
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
    }
    for prop, value in props.items():
        setattr(type(mock), prop, PropertyMock(return_value=value))

    return mock


def _already_installed_mock() -> MagicMock:
    return get_installation_mock(
        name="already_installed",
        already_installed=True,
    )


def _ready_for_installation_mock() -> MagicMock:
    return get_installation_mock(
        name="ready_for_installation",
        already_installed=False,
        could_be_installed=True,
    )


def _could_be_downloaded_and_installed_mock() -> MagicMock:
    return get_installation_mock(
        name="could_be_downloaded_and_installed",
        already_installed=False,
        could_be_installed=True,
        supported_install_type=DownloadAndInstall,
    )


def _could_be_installed_from_mock() -> MagicMock:
    return get_installation_mock(
        name="could_be_installed_from",
        already_installed=False,
        could_be_installed=True,
        supported_install_type=InstallFromPath,
    )


def get_installation_manager(
    noninteractive: bool,
    installations: List[Any],
    monkeypatch: pytest.MonkeyPatch,
    yes_response: bool = True,
) -> DefaultInstallationManager:
    """Get installation manager instance."""
    if not noninteractive:
        monkeypatch.setattr(
            "mlia.tools.metadata.common.yes", MagicMock(return_value=yes_response)
        )

    return DefaultInstallationManager(installations, noninteractive=noninteractive)


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
    assert manager.already_installed() == [already_installed]
    assert manager.ready_for_installation() == [
        ready_for_installation,
        could_be_downloaded_and_installed,
    ]
    assert manager.could_be_downloaded_and_installed() == [
        could_be_downloaded_and_installed
    ]
    assert manager.could_be_downloaded_and_installed("some_installation") == []


@pytest.mark.parametrize("noninteractive", [True, False])
@pytest.mark.parametrize(
    "install_mock, eula_agreement, backend_name, expected_call",
    [
        [
            _could_be_downloaded_and_installed_mock(),
            True,
            None,
            [call(DownloadAndInstall(eula_agreement=True))],
        ],
        [
            _could_be_downloaded_and_installed_mock(),
            False,
            None,
            [call(DownloadAndInstall(eula_agreement=False))],
        ],
        [
            _could_be_downloaded_and_installed_mock(),
            False,
            "unknown",
            [],
        ],
    ],
)
def test_installation_manager_download_and_install(
    install_mock: MagicMock,
    noninteractive: bool,
    eula_agreement: bool,
    backend_name: Optional[str],
    expected_call: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test installation process."""
    install_mock.reset_mock()

    manager = get_installation_manager(noninteractive, [install_mock], monkeypatch)

    manager.download_and_install(backend_name, eula_agreement=eula_agreement)
    assert install_mock.install.mock_calls == expected_call


@pytest.mark.parametrize("noninteractive", [True, False])
@pytest.mark.parametrize(
    "install_mock, backend_name, expected_call",
    [
        [
            _could_be_installed_from_mock(),
            None,
            [call(InstallFromPath(Path("some_path")))],
        ],
        [
            _could_be_installed_from_mock(),
            "unknown",
            [],
        ],
        [
            _already_installed_mock(),
            "already_installed",
            [],
        ],
    ],
)
def test_installation_manager_install_from(
    install_mock: MagicMock,
    noninteractive: bool,
    backend_name: Optional[str],
    expected_call: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test installation process."""
    install_mock.reset_mock()

    manager = get_installation_manager(noninteractive, [install_mock], monkeypatch)
    manager.install_from(Path("some_path"), backend_name)

    assert install_mock.install.mock_calls == expected_call
