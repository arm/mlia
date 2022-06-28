# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""System backend module."""
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from mlia.backend.common import Backend
from mlia.backend.common import ConfigurationException
from mlia.backend.common import get_backend_configs
from mlia.backend.common import get_backend_directories
from mlia.backend.common import load_config
from mlia.backend.common import remove_backend
from mlia.backend.config import SystemConfig
from mlia.backend.controller import SystemController
from mlia.backend.controller import SystemControllerSingleInstance
from mlia.backend.fs import get_backends_path
from mlia.backend.protocol import ProtocolFactory
from mlia.backend.protocol import SupportsClose
from mlia.backend.protocol import SupportsConnection
from mlia.backend.protocol import SupportsDeploy
from mlia.backend.source import create_destination_and_install
from mlia.backend.source import get_source


def get_available_systems_directory_names() -> List[str]:
    """Return a list of directory names for all avialable systems."""
    return [entry.name for entry in get_backend_directories("systems")]


def get_available_systems() -> List["System"]:
    """Return a list with all available systems."""
    available_systems = []
    for config_json in get_backend_configs("systems"):
        config_entries = cast(List[SystemConfig], (load_config(config_json)))
        for config_entry in config_entries:
            config_entry["config_location"] = config_json.parent.absolute()
            system = load_system(config_entry)
            available_systems.append(system)

    return sorted(available_systems, key=lambda system: system.name)


def get_system(system_name: str) -> Optional["System"]:
    """Return a system instance with the same name passed as argument."""
    available_systems = get_available_systems()
    for system in available_systems:
        if system_name == system.name:
            return system
    return None


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
            "Systems [{}] are already installed".format(",".join(names))
        )

    create_destination_and_install(source, get_backends_path("systems"))


def remove_system(directory_name: str) -> None:
    """Remove system."""
    remove_backend(directory_name, "systems")


class System(Backend):
    """System class."""

    def __init__(self, config: SystemConfig) -> None:
        """Construct the System class using the dictionary passed."""
        super().__init__(config)

        self._setup_data_transfer(config)
        self._setup_reporting(config)

    def _setup_data_transfer(self, config: SystemConfig) -> None:
        data_transfer_config = config.get("data_transfer")
        protocol = ProtocolFactory().get_protocol(
            data_transfer_config, cwd=self.config_location
        )
        self.protocol = protocol

    def _setup_reporting(self, config: SystemConfig) -> None:
        self.reporting = config.get("reporting")

    def run(self, command: str, retry: bool = True) -> Tuple[int, bytearray, bytearray]:
        """
        Run command on the system.

        Returns a tuple: (exit_code, stdout, stderr)
        """
        return self.protocol.run(command, retry)

    def deploy(self, src: Path, dst: str, retry: bool = True) -> None:
        """Deploy files to the system."""
        if isinstance(self.protocol, SupportsDeploy):
            self.protocol.deploy(src, dst, retry)

    @property
    def supports_deploy(self) -> bool:
        """Check if protocol supports deploy operation."""
        return isinstance(self.protocol, SupportsDeploy)

    @property
    def connectable(self) -> bool:
        """Check if protocol supports connection."""
        return isinstance(self.protocol, SupportsConnection)

    def establish_connection(self) -> bool:
        """Establish connection with the system."""
        if not isinstance(self.protocol, SupportsConnection):
            raise ConfigurationException(
                "System {} does not support connections".format(self.name)
            )

        return self.protocol.establish_connection()

    def connection_details(self) -> Tuple[str, int]:
        """Return connection details."""
        if not isinstance(self.protocol, SupportsConnection):
            raise ConfigurationException(
                "System {} does not support connections".format(self.name)
            )

        return self.protocol.connection_details()

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, System):
            return False

        return super().__eq__(other) and self.name == other.name

    def get_details(self) -> Dict[str, Any]:
        """Return a dictionary with all relevant information of a System."""
        output = {
            "type": "system",
            "name": self.name,
            "description": self.description,
            "data_transfer_protocol": self.protocol.protocol,
            "commands": self._get_command_details(),
            "annotations": self.annotations,
        }

        return output


class StandaloneSystem(System):
    """StandaloneSystem class."""


def get_controller(
    single_instance: bool, pid_file_path: Optional[Path] = None
) -> SystemController:
    """Get system controller."""
    if single_instance:
        return SystemControllerSingleInstance(pid_file_path)

    return SystemController()


class ControlledSystem(System):
    """ControlledSystem class."""

    def __init__(self, config: SystemConfig):
        """Construct the ControlledSystem class using the dictionary passed."""
        super().__init__(config)
        self.controller: Optional[SystemController] = None

    def start(
        self,
        commands: List[str],
        single_instance: bool = True,
        pid_file_path: Optional[Path] = None,
    ) -> None:
        """Launch the system."""
        if (
            not isinstance(self.config_location, Path)
            or not self.config_location.is_dir()
        ):
            raise ConfigurationException(
                "System {} has wrong config location".format(self.name)
            )

        self.controller = get_controller(single_instance, pid_file_path)
        self.controller.start(commands, self.config_location)

    def is_running(self) -> bool:
        """Check if system is running."""
        if not self.controller:
            return False

        return self.controller.is_running()

    def get_output(self) -> Tuple[str, str]:
        """Return system output."""
        if not self.controller:
            return "", ""

        return self.controller.get_output()

    def stop(self, wait: bool = False) -> None:
        """Stop the system."""
        if not self.controller:
            raise Exception("System has not been started")

        if isinstance(self.protocol, SupportsClose):
            try:
                self.protocol.close()
            except Exception as error:  # pylint: disable=broad-except
                print(error)
        self.controller.stop(wait)


def load_system(config: SystemConfig) -> Union[StandaloneSystem, ControlledSystem]:
    """Load system based on it's execution type."""
    data_transfer = config.get("data_transfer", {})
    protocol = data_transfer.get("protocol")
    populate_shared_params(config)

    if protocol == "ssh":
        return ControlledSystem(config)

    if protocol == "local":
        return StandaloneSystem(config)

    raise ConfigurationException(
        "Unsupported execution type for protocol {}".format(protocol)
    )


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
    for cmd_name in ["build", "run"]:
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
                    "All parameters for command {} should have aliases".format(cmd_name)
                )
            merged_by_alias = {
                **{p.get("alias"): p for p in shared_user_params},
                **{p.get("alias"): p for p in cmd_user_params},
            }
            cmd_user_params = list(merged_by_alias.values())

        user_params[cmd_name] = cmd_user_params

    config["commands"] = commands
    del user_params["shared"]
