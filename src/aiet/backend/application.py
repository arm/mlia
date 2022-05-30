# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Application backend module."""
import re
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional

from aiet.backend.common import Backend
from aiet.backend.common import ConfigurationException
from aiet.backend.common import DataPaths
from aiet.backend.common import get_backend_configs
from aiet.backend.common import get_backend_directories
from aiet.backend.common import load_application_or_tool_configs
from aiet.backend.common import load_config
from aiet.backend.common import remove_backend
from aiet.backend.config import ApplicationConfig
from aiet.backend.config import ExtendedApplicationConfig
from aiet.backend.source import create_destination_and_install
from aiet.backend.source import get_source
from aiet.utils.fs import get_resources


def get_available_application_directory_names() -> List[str]:
    """Return a list of directory names for all available applications."""
    return [entry.name for entry in get_backend_directories("applications")]


def get_available_applications() -> List["Application"]:
    """Return a list with all available applications."""
    available_applications = []
    for config_json in get_backend_configs("applications"):
        config_entries = cast(List[ExtendedApplicationConfig], load_config(config_json))
        for config_entry in config_entries:
            config_entry["config_location"] = config_json.parent.absolute()
            applications = load_applications(config_entry)
            available_applications += applications

    return sorted(available_applications, key=lambda application: application.name)


def get_application(
    application_name: str, system_name: Optional[str] = None
) -> List["Application"]:
    """Return a list of application instances with provided name."""
    return [
        application
        for application in get_available_applications()
        if application.name == application_name
        and (not system_name or application.can_run_on(system_name))
    ]


def install_application(source_path: Path) -> None:
    """Install application."""
    try:
        source = get_source(source_path)
        config = cast(List[ExtendedApplicationConfig], source.config())
        applications_to_install = [
            s for entry in config for s in load_applications(entry)
        ]
    except Exception as error:
        raise ConfigurationException("Unable to read application definition") from error

    if not applications_to_install:
        raise ConfigurationException("No application definition found")

    available_applications = get_available_applications()
    already_installed = [
        s for s in applications_to_install if s in available_applications
    ]
    if already_installed:
        names = {application.name for application in already_installed}
        raise ConfigurationException(
            "Applications [{}] are already installed".format(",".join(names))
        )

    create_destination_and_install(source, get_resources("applications"))


def remove_application(directory_name: str) -> None:
    """Remove application directory."""
    remove_backend(directory_name, "applications")


def get_unique_application_names(system_name: Optional[str] = None) -> List[str]:
    """Extract a list of unique application names of all application available."""
    return list(
        set(
            application.name
            for application in get_available_applications()
            if not system_name or application.can_run_on(system_name)
        )
    )


class Application(Backend):
    """Class for representing a single application component."""

    def __init__(self, config: ApplicationConfig) -> None:
        """Construct a Application instance from a dict."""
        super().__init__(config)

        self.supported_systems = config.get("supported_systems", [])
        self.deploy_data = config.get("deploy_data", [])

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, Application):
            return False

        return (
            super().__eq__(other)
            and self.name == other.name
            and set(self.supported_systems) == set(other.supported_systems)
        )

    def can_run_on(self, system_name: str) -> bool:
        """Check if the application can run on the system passed as argument."""
        return system_name in self.supported_systems

    def get_deploy_data(self) -> List[DataPaths]:
        """Validate and return data specified in the config file."""
        if self.config_location is None:
            raise ConfigurationException(
                "Unable to get application {} config location".format(self.name)
            )

        deploy_data = []
        for item in self.deploy_data:
            src, dst = item
            src_full_path = self.config_location / src
            assert src_full_path.exists(), "{} does not exists".format(src_full_path)
            deploy_data.append(DataPaths(src_full_path, dst))
        return deploy_data

    def get_details(self) -> Dict[str, Any]:
        """Return dictionary with information about the Application instance."""
        output = {
            "type": "application",
            "name": self.name,
            "description": self.description,
            "supported_systems": self.supported_systems,
            "commands": self._get_command_details(),
        }

        return output

    def remove_unused_params(self) -> None:
        """Remove unused params in commands.

        After merging default and system related configuration application
        could have parameters that are not being used in commands. They
        should be removed.
        """
        for command in self.commands.values():
            indexes_or_aliases = [
                m
                for cmd_str in command.command_strings
                for m in re.findall(r"{user_params:(?P<index_or_alias>\w+)}", cmd_str)
            ]

            only_aliases = all(not item.isnumeric() for item in indexes_or_aliases)
            if only_aliases:
                used_params = [
                    param
                    for param in command.params
                    if param.alias in indexes_or_aliases
                ]
                command.params = used_params


def load_applications(config: ExtendedApplicationConfig) -> List[Application]:
    """Load application.

    Application configuration could contain different parameters/commands for different
    supported systems. For each supported system this function will return separate
    Application instance with appropriate configuration.
    """
    configs = load_application_or_tool_configs(config, ApplicationConfig)
    applications = [Application(cfg) for cfg in configs]
    for application in applications:
        application.remove_unused_params()
    return applications
