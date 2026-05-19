# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend auto-install behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend import manager as backend_manager
from mlia.backend.config import BackendConfiguration, BackendType
from mlia.core.errors import ConfigurationError


def _cfg(
    installation: Any,
    backend_type: BackendType = BackendType.WHEEL,
) -> BackendConfiguration:
    return BackendConfiguration(
        supported_advice=[],
        supported_systems=None,
        backend_type=backend_type,
        installation=installation,
        selectable=True,
    )


def _install(
    name: str,
    installed: bool,
    requires_eula: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        already_installed=installed,
        requires_eula=requires_eula,
        dependencies=[],
    )


def _setup_registry(monkeypatch: pytest.MonkeyPatch, items: dict[str, Any]) -> None:
    monkeypatch.setattr(backend_manager, "ensure_backend_plugins_loaded", lambda: None)
    monkeypatch.setattr(backend_manager.backend_registry, "items", items)


def test_auto_install_non_eula_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _install("vela", installed=False)
    _setup_registry(monkeypatch, {"vela": _cfg(installation)})

    noninteractive_mgr = MagicMock()
    interactive_mgr = MagicMock()

    def _get_manager(noninteractive: bool = False) -> MagicMock:
        return noninteractive_mgr if noninteractive else interactive_mgr

    monkeypatch.setattr(backend_manager, "get_installation_manager", _get_manager)

    backend_manager.ensure_backends_installed(["vela"], accept_eula=None)

    noninteractive_mgr.download_and_install.assert_called_once_with(
        ["vela"], eula_agreement=True, force=False
    )
    interactive_mgr.download_and_install.assert_not_called()


def test_auto_install_eula_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _install("corstone-300", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"corstone-300": _cfg(installation)})

    noninteractive_mgr = MagicMock()
    interactive_mgr = MagicMock()

    def _get_manager(noninteractive: bool = False) -> MagicMock:
        return noninteractive_mgr if noninteractive else interactive_mgr

    monkeypatch.setattr(backend_manager, "get_installation_manager", _get_manager)

    backend_manager.ensure_backends_installed(["corstone-300"], accept_eula=None)

    interactive_mgr.download_and_install.assert_called_once_with(
        ["corstone-300"], eula_agreement=False, force=False
    )
    noninteractive_mgr.download_and_install.assert_not_called()


def test_auto_install_eula_api_requires_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _install("corstone-300", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"corstone-300": _cfg(installation)})

    get_installation_manager = MagicMock()
    monkeypatch.setattr(
        backend_manager, "get_installation_manager", get_installation_manager
    )

    with pytest.raises(ConfigurationError):
        backend_manager.ensure_backends_installed(["corstone-300"], accept_eula=False)

    get_installation_manager.assert_not_called()


def test_auto_install_eula_api_fails_before_non_eula_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vela = _install("vela", installed=False)
    corstone = _install("corstone-300", installed=False, requires_eula=True)
    _setup_registry(
        monkeypatch,
        {
            "vela": _cfg(vela),
            "corstone-300": _cfg(corstone),
        },
    )

    manager = MagicMock()
    monkeypatch.setattr(
        backend_manager,
        "get_installation_manager",
        lambda noninteractive=False: manager,
    )

    with pytest.raises(ConfigurationError):
        backend_manager.ensure_backends_installed(
            ["vela", "corstone-300"], accept_eula=False
        )

    manager.download_and_install.assert_not_called()


def test_auto_install_checks_eula_dependencies_before_installing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _install("frontend", installed=False)
    dependency = _install("dependency", installed=False, requires_eula=True)
    frontend.dependencies = ["dependency"]
    _setup_registry(
        monkeypatch,
        {
            "frontend": _cfg(frontend),
            "dependency": _cfg(dependency),
        },
    )

    manager = MagicMock()
    monkeypatch.setattr(
        backend_manager,
        "get_installation_manager",
        lambda noninteractive=False: manager,
    )

    with pytest.raises(ConfigurationError):
        backend_manager.ensure_backends_installed(["frontend"], accept_eula=False)

    manager.download_and_install.assert_not_called()


def test_auto_install_eula_dependency_uses_interactive_cli_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _install("frontend", installed=False)
    dependency = _install("dependency", installed=False, requires_eula=True)
    frontend.dependencies = ["dependency"]
    _setup_registry(
        monkeypatch,
        {
            "frontend": _cfg(frontend),
            "dependency": _cfg(dependency),
        },
    )

    noninteractive_mgr = MagicMock()
    interactive_mgr = MagicMock()

    def _get_manager(noninteractive: bool = False) -> MagicMock:
        return noninteractive_mgr if noninteractive else interactive_mgr

    monkeypatch.setattr(backend_manager, "get_installation_manager", _get_manager)
    interactive_mgr.download_and_install.side_effect = lambda *_args, **_kwargs: (
        setattr(dependency, "already_installed", True)
    )

    backend_manager.ensure_backends_installed(["frontend"], accept_eula=None)

    interactive_mgr.download_and_install.assert_called_once_with(
        ["dependency"], eula_agreement=False, force=False
    )
    noninteractive_mgr.download_and_install.assert_called_once_with(
        ["frontend"], eula_agreement=True, force=False
    )


def test_auto_install_eula_dependency_cancel_stops_non_eula_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _install("frontend", installed=False)
    dependency = _install("dependency", installed=False, requires_eula=True)
    frontend.dependencies = ["dependency"]
    _setup_registry(
        monkeypatch,
        {
            "frontend": _cfg(frontend),
            "dependency": _cfg(dependency),
        },
    )

    noninteractive_mgr = MagicMock()
    interactive_mgr = MagicMock()

    def _get_manager(noninteractive: bool = False) -> MagicMock:
        return noninteractive_mgr if noninteractive else interactive_mgr

    monkeypatch.setattr(backend_manager, "get_installation_manager", _get_manager)

    backend_manager.ensure_backends_installed(["frontend"], accept_eula=None)

    interactive_mgr.download_and_install.assert_called_once_with(
        ["dependency"], eula_agreement=False, force=False
    )
    noninteractive_mgr.download_and_install.assert_not_called()


def test_auto_install_eula_api_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _install("corstone-300", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"corstone-300": _cfg(installation)})

    noninteractive_mgr = MagicMock()
    monkeypatch.setattr(
        backend_manager,
        "get_installation_manager",
        lambda noninteractive=False: noninteractive_mgr,
    )

    backend_manager.ensure_backends_installed(["corstone-300"], accept_eula=True)

    noninteractive_mgr.download_and_install.assert_called_once_with(
        ["corstone-300"], eula_agreement=True, force=False
    )


def test_auto_install_skips_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_registry(
        monkeypatch, {"builtin": _cfg(None, backend_type=BackendType.BUILTIN)}
    )

    noninteractive_mgr = MagicMock()
    monkeypatch.setattr(
        backend_manager,
        "get_installation_manager",
        lambda noninteractive=False: noninteractive_mgr,
    )

    backend_manager.ensure_backends_installed(["builtin"], accept_eula=None)

    noninteractive_mgr.download_and_install.assert_not_called()


def test_auto_install_rejects_unregistered_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_registry(monkeypatch, {})

    with pytest.raises(ConfigurationError, match="not registered"):
        backend_manager.ensure_backends_installed(["missing"], accept_eula=None)


def test_auto_install_rejects_backend_without_install_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_registry(monkeypatch, {"custom": _cfg(None)})

    with pytest.raises(ConfigurationError, match="installation metadata"):
        backend_manager.ensure_backends_installed(["custom"], accept_eula=None)
