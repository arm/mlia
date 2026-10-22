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
    installation = _install("backend-filler", installed=False)
    _setup_registry(monkeypatch, {"backend-filler": _cfg(installation)})

    noninteractive_mgr = MagicMock()
    interactive_mgr = MagicMock()

    def _get_manager(noninteractive: bool = False) -> MagicMock:
        return noninteractive_mgr if noninteractive else interactive_mgr

    monkeypatch.setattr(backend_manager, "get_installation_manager", _get_manager)

    prompt = MagicMock()
    backend_manager.ensure_backends_installed(
        ["backend-filler"], accept_eula=None, prompt=prompt
    )

    noninteractive_mgr.download_and_install.assert_called_once_with(
        ["backend-filler"], eula_agreement=True, force=False
    )
    interactive_mgr.download_and_install.assert_not_called()
    prompt.assert_not_called()


def test_auto_install_eula_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _install("backend-sample", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"backend-sample": _cfg(installation)})

    noninteractive_mgr = MagicMock()
    interactive_mgr = MagicMock()

    def _get_manager(noninteractive: bool = False) -> MagicMock:
        return noninteractive_mgr if noninteractive else interactive_mgr

    monkeypatch.setattr(backend_manager, "get_installation_manager", _get_manager)

    backend_manager.ensure_backends_installed(["backend-sample"], accept_eula=None)

    interactive_mgr.download_and_install.assert_called_once_with(
        ["backend-sample"], eula_agreement=False, force=False
    )
    noninteractive_mgr.download_and_install.assert_not_called()


def test_auto_install_eula_api_requires_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = _install("backend-sample", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"backend-sample": _cfg(installation)})

    get_installation_manager = MagicMock()
    monkeypatch.setattr(
        backend_manager, "get_installation_manager", get_installation_manager
    )

    prompt = MagicMock(return_value="yes")
    with pytest.raises(ConfigurationError, match="Pass accept_eula=True"):
        backend_manager.ensure_backends_installed(
            ["backend-sample"], accept_eula=False, prompt=prompt
        )

    get_installation_manager.assert_not_called()
    prompt.assert_not_called()


def test_auto_install_eula_api_fails_before_non_eula_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bk_filler = _install("backend-filler", installed=False)
    bk_sample = _install("backend-sample", installed=False, requires_eula=True)
    _setup_registry(
        monkeypatch,
        {
            "backend-filler": _cfg(bk_filler),
            "backend-sample": _cfg(bk_sample),
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
            ["backend-filler", "backend-sample"], accept_eula=False
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
    installation = _install("backend-sample", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"backend-sample": _cfg(installation)})

    noninteractive_mgr = MagicMock()
    monkeypatch.setattr(
        backend_manager,
        "get_installation_manager",
        lambda noninteractive=False: noninteractive_mgr,
    )

    prompt = MagicMock(return_value="no")
    backend_manager.ensure_backends_installed(
        ["backend-sample"], accept_eula=True, prompt=prompt
    )

    noninteractive_mgr.download_and_install.assert_called_once_with(
        ["backend-sample"], eula_agreement=True, force=False
    )
    prompt.assert_not_called()


def test_auto_install_prompt_acceptance_avoids_terminal_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frontend consent selects the noninteractive installation path."""
    installation = _install("backend-sample", installed=False, requires_eula=True)
    _setup_registry(monkeypatch, {"backend-sample": _cfg(installation)})
    get_manager = MagicMock()
    monkeypatch.setattr(backend_manager, "get_installation_manager", get_manager)
    prompt = MagicMock(return_value=" YES ")

    backend_manager.ensure_backends_installed(
        ["backend-sample"], accept_eula=None, prompt=prompt
    )

    prompt.assert_not_called()
    get_manager.assert_called_once_with(noninteractive=True)
    get_manager.return_value.download_and_install.assert_called_once_with(
        ["backend-sample"],
        eula_agreement=False,
        force=False,
        eula_prompt=prompt,
    )


@pytest.mark.parametrize("response", ["no", ""], ids=["refused", "default"])
def test_auto_install_passes_frontend_prompt_to_eula_installation(
    monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    """The staged installer, not the manager, obtains frontend consent."""
    _setup_registry(
        monkeypatch,
        {
            "backend-filler": _cfg(_install("backend-filler", installed=False)),
            "backend-sample": _cfg(
                _install("backend-sample", installed=False, requires_eula=True)
            ),
        },
    )
    get_manager = MagicMock()
    monkeypatch.setattr(backend_manager, "get_installation_manager", get_manager)
    prompt = MagicMock(return_value=response)

    backend_manager.ensure_backends_installed(
        ["backend-filler", "backend-sample"], accept_eula=None, prompt=prompt
    )

    prompt.assert_not_called()
    get_manager.assert_any_call(noninteractive=True)
    get_manager.return_value.download_and_install.assert_any_call(
        ["backend-sample"],
        eula_agreement=False,
        force=False,
        eula_prompt=prompt,
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
