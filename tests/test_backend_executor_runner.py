# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module backend/manager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

from mlia.backend.corstone.performance import BackendRunner
from mlia.backend.corstone.performance import ExecutionParams


class TestBackendRunner:
    """Tests for BackendRunner class."""

    @staticmethod
    def _setup_backends(
        monkeypatch: pytest.MonkeyPatch,
        available_systems: list[str] | None = None,
        available_apps: list[str] | None = None,
    ) -> None:
        """Set up backend metadata."""

        def mock_system(system: str) -> MagicMock:
            """Mock the System instance."""
            mock = MagicMock()
            type(mock).name = PropertyMock(return_value=system)
            return mock

        def mock_app(app: str) -> MagicMock:
            """Mock the Application instance."""
            mock = MagicMock()
            type(mock).name = PropertyMock(return_value=app)
            mock.can_run_on.return_value = True
            return mock

        system_mocks = [mock_system(name) for name in (available_systems or [])]
        monkeypatch.setattr(
            "mlia.backend.executor.runner.get_available_systems",
            MagicMock(return_value=system_mocks),
        )

        apps_mock = [mock_app(name) for name in (available_apps or [])]
        monkeypatch.setattr(
            "mlia.backend.executor.runner.get_available_applications",
            MagicMock(return_value=apps_mock),
        )

    @pytest.mark.parametrize(
        "available_systems, system, installed",
        [
            ([], "system1", False),
            (["system1", "system2"], "system1", True),
        ],
    )
    def test_is_system_installed(
        self,
        available_systems: list,
        system: str,
        installed: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method is_system_installed."""
        backend_runner = BackendRunner()

        self._setup_backends(monkeypatch, available_systems)

        assert backend_runner.is_system_installed(system) == installed

    @pytest.mark.parametrize(
        "available_systems, systems",
        [
            ([], []),
            (["system1"], ["system1"]),
        ],
    )
    def test_installed_systems(
        self,
        available_systems: list[str],
        systems: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method installed_systems."""
        backend_runner = BackendRunner()

        self._setup_backends(monkeypatch, available_systems)
        assert backend_runner.get_installed_systems() == systems

    @staticmethod
    def test_install_system(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test system installation."""
        install_system_mock = MagicMock()
        monkeypatch.setattr(
            "mlia.backend.executor.runner.install_system", install_system_mock
        )

        backend_runner = BackendRunner()
        backend_runner.install_system(Path("test_system_path"))

        install_system_mock.assert_called_once_with(Path("test_system_path"))

    @pytest.mark.parametrize(
        "available_systems, systems, expected_result",
        [
            ([], [], False),
            (["system1"], [], False),
            (["system1"], ["system1"], True),
            (["system1", "system2"], ["system1", "system3"], False),
            (["system1", "system2"], ["system1", "system2"], True),
        ],
    )
    def test_systems_installed(
        self,
        available_systems: list[str],
        systems: list[str],
        expected_result: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method systems_installed."""
        self._setup_backends(monkeypatch, available_systems)

        backend_runner = BackendRunner()

        assert backend_runner.systems_installed(systems) is expected_result

    @pytest.mark.parametrize(
        "available_apps, applications, expected_result",
        [
            ([], [], False),
            (["app1"], [], False),
            (["app1"], ["app1"], True),
            (["app1", "app2"], ["app1", "app3"], False),
            (["app1", "app2"], ["app1", "app2"], True),
        ],
    )
    def test_applications_installed(
        self,
        available_apps: list[str],
        applications: list[str],
        expected_result: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method applications_installed."""
        self._setup_backends(monkeypatch, [], available_apps)
        backend_runner = BackendRunner()

        assert backend_runner.applications_installed(applications) is expected_result

    @pytest.mark.parametrize(
        "available_apps, applications",
        [
            ([], []),
            (
                ["application1", "application2"],
                ["application1", "application2"],
            ),
        ],
    )
    def test_get_installed_applications(
        self,
        available_apps: list[str],
        applications: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method get_installed_applications."""
        self._setup_backends(monkeypatch, [], available_apps)

        backend_runner = BackendRunner()
        assert applications == backend_runner.get_installed_applications()

    @staticmethod
    def test_install_application(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test application installation."""
        mock_install_application = MagicMock()
        monkeypatch.setattr(
            "mlia.backend.executor.runner.install_application",
            mock_install_application,
        )

        backend_runner = BackendRunner()
        backend_runner.install_application(Path("test_application_path"))
        mock_install_application.assert_called_once_with(Path("test_application_path"))

    @pytest.mark.parametrize(
        "available_apps, application, installed",
        [
            ([], "system1", False),
            (
                ["application1", "application2"],
                "application1",
                True,
            ),
            (
                [],
                "application1",
                False,
            ),
        ],
    )
    def test_is_application_installed(
        self,
        available_apps: list[str],
        application: str,
        installed: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method is_application_installed."""
        self._setup_backends(monkeypatch, [], available_apps)

        backend_runner = BackendRunner()
        assert installed == backend_runner.is_application_installed(
            application, "system1"
        )

    @staticmethod
    @pytest.mark.parametrize(
        "execution_params, expected_command",
        [
            (
                ExecutionParams("application_4", "System 4", [], []),
                ["application_4", [], "System 4", []],
            ),
            (
                ExecutionParams(
                    "application_6",
                    "System 6",
                    ["param1=value2"],
                    ["sys-param1=value2"],
                ),
                [
                    "application_6",
                    ["param1=value2"],
                    "System 6",
                    ["sys-param1=value2"],
                ],
            ),
        ],
    )
    def test_run_application_local(
        monkeypatch: pytest.MonkeyPatch,
        execution_params: ExecutionParams,
        expected_command: list[str],
    ) -> None:
        """Test method run_application with local systems."""
        run_app = MagicMock()
        monkeypatch.setattr("mlia.backend.executor.runner.run_application", run_app)

        backend_runner = BackendRunner()
        backend_runner.run_application(execution_params)

        run_app.assert_called_once_with(*expected_command)
