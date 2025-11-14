# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for installation process."""
from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable

from mlia.backend.config import BackendType
from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import Installation
from mlia.backend.install import InstallationType
from mlia.backend.install import InstallFromPath
from mlia.backend.registry import registry as backend_registry
from mlia.core.errors import ConfigurationError
from mlia.core.errors import InternalError
from mlia.utils.misc import yes


logger = logging.getLogger(__name__)

InstallationFilter = Callable[[Installation], bool]


class AlreadyInstalledFilter:
    """Filter for already installed backends."""

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return installation.already_installed


class ReadyForInstallationFilter:
    """Filter for ready to be installed backends."""

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return installation.could_be_installed and not installation.already_installed


class SupportsInstallTypeFilter:
    """Filter backends that support certain type of the installation."""

    def __init__(self, installation_type: InstallationType) -> None:
        """Init filter."""
        self.installation_type = installation_type

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return installation.supports(self.installation_type)


class SearchByNameFilter:
    """Filter installation by name."""

    def __init__(self, backend_name: str | None) -> None:
        """Init filter."""
        self.backend_name = backend_name

    def __call__(self, installation: Installation) -> bool:
        """Installation filter."""
        return (
            not self.backend_name
            or installation.name.casefold() == self.backend_name.casefold()
        )


class InstallationManager(ABC):
    """Helper class for managing installations."""

    @abstractmethod
    def install_from(self, backend_path: Path, backend_name: str, force: bool) -> None:
        """Install backend from the local directory."""

    @abstractmethod
    def download_and_install(
        self, backend_names: list[str], eula_agreement: bool, force: bool
    ) -> None:
        """Download and install backends."""

    @abstractmethod
    def show_env_details(self) -> None:
        """Show environment details."""

    @abstractmethod
    def backend_installed(self, backend_name: str) -> bool:
        """Return true if requested backend installed."""

    @abstractmethod
    def uninstall(self, backend_names: list[str]) -> None:
        """Delete the existing installation."""


class InstallationFiltersMixin:
    """Mixin for filtering installation based on different conditions."""

    installations: list[Installation]

    def filter_by(self, *filters: InstallationFilter) -> list[Installation]:
        """Filter installations."""
        return [
            installation
            for installation in self.installations
            if all(filter_(installation) for filter_ in filters)
        ]

    def find_by_name(self, backend_name: str) -> list[Installation]:
        """Return list of the backends filtered by name."""
        return self.filter_by(SearchByNameFilter(backend_name))

    def already_installed(self, backend_name: str | None = None) -> list[Installation]:
        """Return list of backends that are already installed."""
        return self.filter_by(
            AlreadyInstalledFilter(),
            SearchByNameFilter(backend_name),
        )

    def ready_for_installation(self) -> list[Installation]:
        """Return list of the backends that could be installed."""
        return self.filter_by(ReadyForInstallationFilter())


class DefaultInstallationManager(InstallationManager, InstallationFiltersMixin):
    """Interactive installation manager."""

    def __init__(
        self, installations: list[Installation], noninteractive: bool = False
    ) -> None:
        """Init the manager."""
        self.installations = installations
        self.noninteractive = noninteractive

    def _resolve_backend(self, name: str, err_msg_prefix: str = "") -> Installation:
        candidate_installs = self.find_by_name(name)
        if not candidate_installs:
            logger.info("Unknown backend '%s'.", name)
            logger.info(
                "Please run command 'mlia-backend list' to get list of "
                "supported backend names."
            )
            raise ValueError(f"{err_msg_prefix}: Could not resolve {name} backend.")

        if len(candidate_installs) > 1:
            raise InternalError(
                f"{err_msg_prefix}: More than one backend with name " f"{name} found."
            )

        return candidate_installs[0]

    def _get_installable_installation(
        self, name: str, install_type: InstallationType, force: bool
    ) -> Installation | None:
        installation = self._resolve_backend(name)

        if installation.already_installed and not force:
            logger.info("Backend '%s' is already installed.", installation.name)
            logger.info("Please, consider using --force option.")
            return None

        if not installation.supports(install_type):
            logger.info(
                "Installation method %s not supported for %s backend.",
                install_type,
                name,
            )
            return None

        return installation

    def _get_dependency_installations(
        self, root_installations: list[Installation]
    ) -> list[Installation]:
        """
        DFS-traverse the dependency graph and list all dependencies.

        ValueError will be raised if a dependency cannot be resolved.
        InternalError will be raise if:
            - a dependency does not support DownloadAndInstall,
            - a circular dependency is detected.
        """
        visits: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        dep_installations: list[Installation] = []
        root_names = {root.name for root in root_installations}

        # DFS traversal that enlists all encountered nodes, but does not
        # enlist root nodes as dependencies.
        for root in root_installations:
            if visits.get(root.name, 0) == 2:
                # already fully processed
                continue

            stack: list[tuple[Installation, int]] = [
                (root, 0)
            ]  # (node, next_dep_index)
            while stack:
                inst, i = stack[-1]
                name = inst.name
                visited = visits.get(name, 0)
                if visited == 0:
                    visits[name] = 1  # entering
                deps = inst.dependencies

                if i < len(deps):
                    dep_name = deps[i]
                    stack[-1] = (inst, i + 1)
                    dep = self._resolve_backend(
                        dep_name, f"Failed to resolve dependencies for {root.name}:"
                    )
                    dep_visits = visits.get(dep.name, 0)
                    if dep_visits == 0:
                        stack.append((dep, 0))
                        # Exclude root packages from the dependencies list
                        if dep.name not in root_names:
                            dep_installations.append(
                                dep
                            )  # preorder: append on discovery
                    elif dep_visits == 1:
                        raise InternalError(
                            f"Dependency cycle detected involving {dep.name}"
                        )
                else:
                    # all deps done; exit node
                    stack.pop()
                    visits[name] = 2

        return dep_installations

    def _install(
        self,
        backend_names: list[str],
        install_types: list[InstallationType],
        force: bool,
    ) -> None:
        """Check metadata and install backend."""
        installations = [
            inst
            for name, type in zip(backend_names, install_types)
            if (inst := self._get_installable_installation(name, type, force))
            is not None
        ]
        if not installations:
            return

        dep_installations = self._get_dependency_installations(installations)

        # Filter out installed dependencies and collect their
        # install types
        dep_installations_to_be_installed = []
        dep_install_types_to_be_installed = []
        for inst in dep_installations:
            if inst.already_installed:
                logger.debug("%s dependency already installed.", inst.name)
                continue
            dep_installations_to_be_installed.append(inst)

            inst_type = self._get_default_insallation_type(inst)
            if inst_type is None:
                raise InternalError(
                    f"{inst.name} found, but can only be installed from a file."
                )
            dep_install_types_to_be_installed.append(inst_type)

        # Show all backends to be installed
        logger.info("Installing the following backends:")
        for inst in installations:
            logger.info("* %s", inst.name)
        if dep_installations_to_be_installed:
            logger.info("Dependencies will be installed automatically:")
            for inst in dep_installations_to_be_installed:
                logger.info("* %s", inst.name)

        if not self.noninteractive and not yes("Would you like to proceed?"):
            return

        # Install dependencies
        for inst, inst_type in zip(
            dep_installations_to_be_installed, dep_install_types_to_be_installed
        ):
            logger.info("Installing %s", inst.name)
            inst.install(inst_type)
            logger.info("%s successfully installed", inst.name)

        # Install backends
        for inst, inst_type in zip(installations, install_types):
            if inst.already_installed and force:
                logger.info(
                    "Force installing %s, so delete the existing "
                    "installed backend first.",
                    inst.name,
                )
                inst.uninstall()
            logger.info("Installing %s", inst.name)
            inst.install(inst_type)
            logger.info("%s successfully installed", inst.name)

    def _get_default_insallation_type(
        self, installation: Installation
    ) -> InstallationType | None:
        if installation.supports(DownloadAndInstall()):
            return DownloadAndInstall()
        return None

    def install_from(
        self, backend_path: Path, backend_name: str, force: bool = False
    ) -> None:
        """Install from the provided directory."""
        self._install([backend_name], [InstallFromPath(backend_path)], force)

    def download_and_install(
        self, backend_names: list[str], eula_agreement: bool = True, force: bool = False
    ) -> None:
        """Download and install available backends."""
        install_types = [DownloadAndInstall(eula_agreement=eula_agreement)] * len(
            backend_names
        )
        self._install(backend_names, install_types, force)  # type: ignore[arg-type]

    def show_env_details(self) -> None:
        """Print current state of the execution environment."""
        if installed := self.already_installed():
            self._print_installation_list("Installed backends:", installed)

        if could_be_installed := self.ready_for_installation():
            self._print_installation_list(
                "Following backends could be installed:",
                could_be_installed,
                new_section=bool(installed),
            )

        if not installed and not could_be_installed:
            logger.info("No backends installed")

    @staticmethod
    def _print_installation_list(
        header: str, installations: list[Installation], new_section: bool = False
    ) -> None:
        """Print list of the installations."""
        logger.info("%s%s\n", "\n" if new_section else "", header)

        for installation in installations:
            logger.info("  - %s", installation.name)

    def uninstall(self, backend_names: list[str]) -> None:
        """Uninstall the backend with name backend_name."""
        for backend_name in backend_names:
            installations = self.already_installed(backend_name)

            if not installations:
                raise ConfigurationError(f"Backend '{backend_name}' is not installed.")

            if len(installations) != 1:
                raise InternalError(
                    f"More than one installed backend with name {backend_name} found."
                )

            installation = installations[0]

            dependent_backends = [
                dep.name
                for dep in self.installations
                if installation.name in dep.dependencies and dep.already_installed
            ]
            if dependent_backends:
                msg = (
                    f"The following backends depend on '{installation.name}' which "
                    f"you are about to uninstall: {dependent_backends}",
                )
                proceed = self.noninteractive or yes(
                    f"{msg}. Continue uninstalling anyway?"
                )
                logger.warning(msg)
                if not proceed:
                    logger.info(
                        "Uninstalling %s canceled due to dependencies.",
                        installation.name,
                    )
                    return

            installation.uninstall()
            logger.info("%s successfully uninstalled.", installation.name)

    def backend_installed(self, backend_name: str) -> bool:
        """Return true if requested backend installed."""
        installations = self.already_installed(backend_name)

        return len(installations) == 1


def get_installation_manager(noninteractive: bool = False) -> InstallationManager:
    """Return installation manager."""
    backends = [
        cfg.installation for cfg in backend_registry.items.values() if cfg.installation
    ]
    return DefaultInstallationManager(backends, noninteractive=noninteractive)


def get_available_backends() -> list[str]:
    """Return list of the available backends."""
    manager = get_installation_manager()
    available_backends = [
        backend
        for backend, cfg in backend_registry.items.items()
        if cfg.type == BackendType.BUILTIN or manager.backend_installed(backend)
    ]
    return available_backends
