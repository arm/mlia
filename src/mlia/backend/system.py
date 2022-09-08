# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""System backend module."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast
from typing import List

from mlia.backend.common import Backend
from mlia.backend.common import ConfigurationException
from mlia.backend.common import get_backend_configs
from mlia.backend.common import get_backend_directories
from mlia.backend.common import load_config
from mlia.backend.common import remove_backend
from mlia.backend.config import SystemConfig
from mlia.backend.fs import get_backends_path
from mlia.backend.proc import run_and_wait
from mlia.backend.source import create_destination_and_install
from mlia.backend.source import get_source


class System(Backend):
    """System class."""

    def __init__(self, config: SystemConfig) -> None:
        """Construct the System class using the dictionary passed."""
        super().__init__(config)

        self._setup_reporting(config)

    def _setup_reporting(self, config: SystemConfig) -> None:
        self.reporting = config.get("reporting")

    def run(self, command: str) -> tuple[int, bytearray, bytearray]:
        """
        Run command on the system.

        Returns a tuple: (exit_code, stdout, stderr)
        """
        cwd = self.config_location
        if not isinstance(cwd, Path) or not cwd.is_dir():
            raise ConfigurationException(
                f"System has invalid config location: {cwd}",
            )

        stdout = bytearray()
        stderr = bytearray()

        return run_and_wait(
            command,
            cwd=cwd,
            terminate_on_error=True,
            out=stdout,
            err=stderr,
        )

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, System):
            return False

        return super().__eq__(other) and self.name == other.name

    def get_details(self) -> dict[str, Any]:
        """Return a dictionary with all relevant information of a System."""
        output = {
            "type": "system",
            "name": self.name,
            "description": self.description,
            "commands": self._get_command_details(),
            "annotations": self.annotations,
        }

        return output


def get_available_systems_directory_names() -> list[str]:
    """Return a list of directory names for all avialable systems."""
    return [entry.name for entry in get_backend_directories("systems")]


def get_available_systems() -> list[System]:
    """Return a list with all available systems."""
    available_systems = []
    for config_json in get_backend_configs("systems"):
        config_entries = cast(List[SystemConfig], (load_config(config_json)))
        for config_entry in config_entries:
            config_entry["config_location"] = config_json.parent.absolute()
            system = load_system(config_entry)
            available_systems.append(system)

    return sorted(available_systems, key=lambda system: system.name)


def get_system(system_name: str) -> System:
    """Return a system instance with the same name passed as argument."""
    available_systems = get_available_systems()
    for system in available_systems:
        if system_name == system.name:
            return system
    raise ConfigurationException(f"System '{system_name}' not found.")


def install_system(source_path: Path) -> None:
    """Install new system."""
    try:
        source = get_source(source_path)
        config = cast(List[SystemConfig], source.config())
        systems_to_install = [load_system(entry) for entry in config]
    except Exception as error:
        raise ConfigurationException("Unable to read system definition") from error

    if not systems_to_install:
        raise ConfigurationException("No system definition found")

    available_systems = get_available_systems()
    already_installed = [s for s in systems_to_install if s in available_systems]
    if already_installed:
        names = [system.name for system in already_installed]
        raise ConfigurationException(
            f"Systems [{','.join(names)}] are already installed."
        )

    create_destination_and_install(source, get_backends_path("systems"))


def remove_system(directory_name: str) -> None:
    """Remove system."""
    remove_backend(directory_name, "systems")


def load_system(config: SystemConfig) -> System:
    """Load system based on it's execution type."""
    populate_shared_params(config)

    return System(config)


def populate_shared_params(config: SystemConfig) -> None:
    """Populate command parameters with shared parameters."""
    user_params = config.get("user_params")
    if not user_params or "shared" not in user_params:
        return

    shared_user_params = user_params["shared"]
    if not shared_user_params:
        return

    only_aliases = all(p.get("alias") for p in shared_user_params)
    if not only_aliases:
        raise ConfigurationException("All shared parameters should have aliases")

    commands = config.get("commands", {})
    for cmd_name in ["run"]:
        command = commands.get(cmd_name)
        if command is None:
            commands[cmd_name] = []
        cmd_user_params = user_params.get(cmd_name)
        if not cmd_user_params:
            cmd_user_params = shared_user_params
        else:
            only_aliases = all(p.get("alias") for p in cmd_user_params)
            if not only_aliases:
                raise ConfigurationException(
                    f"All parameters for command {cmd_name} should have aliases."
                )
            merged_by_alias = {
                **{p.get("alias"): p for p in shared_user_params},
                **{p.get("alias"): p for p in cmd_user_params},
            }
            cmd_user_params = list(merged_by_alias.values())

        user_params[cmd_name] = cmd_user_params

    config["commands"] = commands
    del user_params["shared"]
