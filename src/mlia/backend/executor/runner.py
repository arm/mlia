# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend runner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlia.backend.executor.application import get_available_applications
from mlia.backend.executor.application import install_application
from mlia.backend.executor.execution import ExecutionContext
from mlia.backend.executor.execution import run_application
from mlia.backend.executor.system import get_available_systems
from mlia.backend.executor.system import install_system


@dataclass
class ExecutionParams:
    """Application execution params."""

    application: str
    system: str
    application_params: list[str]
    system_params: list[str]


class BackendRunner:
    """Backend runner."""

    def __init__(self) -> None:
        """Init BackendRunner instance."""

    @staticmethod
    def get_installed_systems() -> list[str]:
        """Get list of the installed systems."""
        return [system.name for system in get_available_systems()]

    @staticmethod
    def get_installed_applications(system: str | None = None) -> list[str]:
        """Get list of the installed application."""
        return [
            app.name
            for app in get_available_applications()
            if system is None or app.can_run_on(system)
        ]

    def is_application_installed(self, application: str, system: str) -> bool:
        """Return true if requested application installed."""
        return application in self.get_installed_applications(system)

    def is_system_installed(self, system: str) -> bool:
        """Return true if requested system installed."""
        return system in self.get_installed_systems()

    def systems_installed(self, systems: list[str]) -> bool:
        """Check if all provided systems are installed."""
        if not systems:
            return False

        installed_systems = self.get_installed_systems()
        return all(system in installed_systems for system in systems)

    def applications_installed(self, applications: list[str]) -> bool:
        """Check if all provided applications are installed."""
        if not applications:
            return False

        installed_apps = self.get_installed_applications()
        return all(app in installed_apps for app in applications)

    def all_installed(self, systems: list[str], apps: list[str]) -> bool:
        """Check if all provided artifacts are installed."""
        return self.systems_installed(systems) and self.applications_installed(apps)

    @staticmethod
    def install_system(system_path: Path) -> None:
        """Install system."""
        install_system(system_path)

    @staticmethod
    def install_application(app_path: Path) -> None:
        """Install application."""
        install_application(app_path)

    @staticmethod
    def run_application(execution_params: ExecutionParams) -> ExecutionContext:
        """Run requested application."""
        ctx = run_application(
            execution_params.application,
            execution_params.application_params,
            execution_params.system,
            execution_params.system_params,
        )
        return ctx

    @staticmethod
    def _params(name: str, params: list[str]) -> list[str]:
        return [p for item in [(name, param) for param in params] for p in item]
