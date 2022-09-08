# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contain all common functions for the backends."""
from __future__ import annotations

import json
import logging
import re
from abc import ABC
from collections import Counter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Final
from typing import IO
from typing import Iterable
from typing import Match
from typing import NamedTuple
from typing import Pattern

from mlia.backend.config import BackendConfig
from mlia.backend.config import BaseBackendConfig
from mlia.backend.config import NamedExecutionConfig
from mlia.backend.config import UserParamConfig
from mlia.backend.config import UserParamsConfig
from mlia.backend.fs import get_backends_path
from mlia.backend.fs import remove_resource
from mlia.backend.fs import ResourceType


BACKEND_CONFIG_FILE: Final[str] = "backend-config.json"


class ConfigurationException(Exception):
    """Configuration exception."""


def get_backend_config(dir_path: Path) -> Path:
    """Get path to backendir configuration file."""
    return dir_path / BACKEND_CONFIG_FILE


def get_backend_configs(resource_type: ResourceType) -> Iterable[Path]:
    """Get path to the backend configs for provided resource_type."""
    return (
        get_backend_config(entry) for entry in get_backend_directories(resource_type)
    )


def get_backend_directories(resource_type: ResourceType) -> Iterable[Path]:
    """Get path to the backend directories for provided resource_type."""
    return (
        entry
        for entry in get_backends_path(resource_type).iterdir()
        if is_backend_directory(entry)
    )


def is_backend_directory(dir_path: Path) -> bool:
    """Check if path is backend's configuration directory."""
    return dir_path.is_dir() and get_backend_config(dir_path).is_file()


def remove_backend(directory_name: str, resource_type: ResourceType) -> None:
    """Remove backend with provided type and directory_name."""
    if not directory_name:
        raise Exception("No directory name provided")

    remove_resource(directory_name, resource_type)


def load_config(config: Path | IO[bytes] | None) -> BackendConfig:
    """Return a loaded json file."""
    if config is None:
        raise Exception("Unable to read config")

    if isinstance(config, Path):
        with config.open() as json_file:
            return cast(BackendConfig, json.load(json_file))

    return cast(BackendConfig, json.load(config))


def parse_raw_parameter(parameter: str) -> tuple[str, str | None]:
    """Split the parameter string in name and optional value.

    It manages the following cases:
    --param=1 -> --param, 1
    --param 1 -> --param, 1
    --flag    -> --flag, None
    """
    data = re.split(" |=", parameter)
    if len(data) == 1:
        param_name = data[0]
        param_value = None
    else:
        param_name = " ".join(data[0:-1])
        param_value = data[-1]
    return param_name, param_value


class DataPaths(NamedTuple):
    """DataPaths class."""

    src: Path
    dst: str


class Backend(ABC):
    """Backend class."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, config: BaseBackendConfig):
        """Initialize backend."""
        name = config.get("name")
        if not name:
            raise ConfigurationException("Name is empty")

        self.name = name
        self.description = config.get("description", "")
        self.config_location = config.get("config_location")
        self.variables = config.get("variables", {})
        self.annotations = config.get("annotations", {})

        self._parse_commands_and_params(config)

    def validate_parameter(self, command_name: str, parameter: str) -> bool:
        """Validate the parameter string against the application configuration.

        We take the parameter string, extract the parameter name/value and
        check them against the current configuration.
        """
        param_name, param_value = parse_raw_parameter(parameter)
        valid_param_name = valid_param_value = False

        command = self.commands.get(command_name)
        if not command:
            raise AttributeError(f"Unknown command: '{command_name}'")

        # Iterate over all available parameters until we have a match.
        for param in command.params:
            if self._same_parameter(param_name, param):
                valid_param_name = True
                # This is a non-empty list
                if param.values:
                    # We check if the value is allowed in the configuration
                    valid_param_value = param_value in param.values
                else:
                    # In this case we don't validate the value and accept
                    # whatever we have set.
                    valid_param_value = True
                break

        return valid_param_name and valid_param_value

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, Backend):
            return False

        return (
            self.name == other.name
            and self.description == other.description
            and self.commands == other.commands
        )

    def __repr__(self) -> str:
        """Represent the Backend instance by its name."""
        return self.name

    def _parse_commands_and_params(self, config: BaseBackendConfig) -> None:
        """Parse commands and user parameters."""
        self.commands: dict[str, Command] = {}

        commands = config.get("commands")
        if commands:
            params = config.get("user_params")

            for command_name in commands.keys():
                command_params = self._parse_params(params, command_name)
                command_strings = [
                    self._substitute_variables(cmd)
                    for cmd in commands.get(command_name, [])
                ]
                self.commands[command_name] = Command(command_strings, command_params)

    def _substitute_variables(self, str_val: str) -> str:
        """Substitute variables in string.

        Variables is being substituted at backend's creation stage because
        they could contain references to other params which will be
        resolved later.
        """
        if not str_val:
            return str_val

        var_pattern: Final[Pattern] = re.compile(r"{variables:(?P<var_name>\w+)}")

        def var_value(match: Match) -> str:
            var_name = match["var_name"]
            if var_name not in self.variables:
                raise ConfigurationException(f"Unknown variable {var_name}")

            return self.variables[var_name]

        return var_pattern.sub(var_value, str_val)  # type: ignore

    @classmethod
    def _parse_params(
        cls, params: UserParamsConfig | None, command: str
    ) -> list[Param]:
        if not params:
            return []

        return [cls._parse_param(p) for p in params.get(command, [])]

    @classmethod
    def _parse_param(cls, param: UserParamConfig) -> Param:
        """Parse a single parameter."""
        name = param.get("name")
        if name is not None and not name:
            raise ConfigurationException("Parameter has an empty 'name' attribute.")
        values = param.get("values", None)
        default_value = param.get("default_value", None)
        description = param.get("description", "")
        alias = param.get("alias")

        return Param(
            name=name,
            description=description,
            values=values,
            default_value=default_value,
            alias=alias,
        )

    def _get_command_details(self) -> dict:
        command_details = {
            command_name: command.get_details()
            for command_name, command in self.commands.items()
        }
        return command_details

    def _get_user_param_value(self, user_params: list[str], param: Param) -> str | None:
        """Get the user-specified value of a parameter."""
        for user_param in user_params:
            user_param_name, user_param_value = parse_raw_parameter(user_param)
            if user_param_name == param.name:
                warn_message = (
                    "The direct use of parameter name is deprecated"
                    " and might be removed in the future.\n"
                    f"Please use alias '{param.alias}' instead of "
                    "'{user_param_name}' to provide the parameter."
                )
                logging.warning(warn_message)

            if self._same_parameter(user_param_name, param):
                return user_param_value

        return None

    @staticmethod
    def _same_parameter(user_param_name_or_alias: str, param: Param) -> bool:
        """Compare user parameter name with param name or alias."""
        # Strip the "=" sign in the param_name. This is needed just for
        # comparison with the parameters passed by the user.
        # The equal sign needs to be honoured when re-building the
        # parameter back.
        param_name = None if not param.name else param.name.rstrip("=")
        return user_param_name_or_alias in [param_name, param.alias]

    def resolved_parameters(
        self, command_name: str, user_params: list[str]
    ) -> list[tuple[str | None, Param]]:
        """Return list of parameters with values."""
        result: list[tuple[str | None, Param]] = []
        command = self.commands.get(command_name)
        if not command:
            return result

        for param in command.params:
            value = self._get_user_param_value(user_params, param)
            if not value:
                value = param.default_value
            result.append((value, param))

        return result

    def build_command(
        self,
        command_name: str,
        user_params: list[str],
        param_resolver: Callable[[str, str, list[tuple[str | None, Param]]], str],
    ) -> list[str]:
        """
        Return a list of executable command strings.

        Given a command and associated parameters, returns a list of executable command
        strings.
        """
        command = self.commands.get(command_name)
        if not command:
            raise ConfigurationException(
                f"Command '{command_name}' could not be found."
            )

        commands_to_run = []

        params_values = self.resolved_parameters(command_name, user_params)
        for cmd_str in command.command_strings:
            cmd_str = resolve_all_parameters(
                cmd_str, param_resolver, command_name, params_values
            )
            commands_to_run.append(cmd_str)

        return commands_to_run


class Param:
    """Class for representing a generic application parameter."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str | None,
        description: str,
        values: list[str] | None = None,
        default_value: str | None = None,
        alias: str | None = None,
    ) -> None:
        """Construct a Param instance."""
        if not name and not alias:
            raise ConfigurationException(
                "Either name, alias or both must be set to identify a parameter."
            )
        self.name = name
        self.values = values
        self.description = description
        self.default_value = default_value
        self.alias = alias

    def get_details(self) -> dict:
        """Return a dictionary with all relevant information of a Param."""
        return {key: value for key, value in self.__dict__.items() if value}

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, Param):
            return False

        return (
            self.name == other.name
            and self.values == other.values
            and self.default_value == other.default_value
            and self.description == other.description
        )


class Command:
    """Class for representing a command."""

    def __init__(
        self, command_strings: list[str], params: list[Param] | None = None
    ) -> None:
        """Construct a Command instance."""
        self.command_strings = command_strings

        if params:
            self.params = params
        else:
            self.params = []

        self._validate()

    def _validate(self) -> None:
        """Validate command."""
        if not self.params:
            return

        aliases = [param.alias for param in self.params if param.alias is not None]
        repeated_aliases = [
            alias for alias, count in Counter(aliases).items() if count > 1
        ]

        if repeated_aliases:
            raise ConfigurationException(
                f"Non-unique aliases {', '.join(repeated_aliases)}"
            )

        both_name_and_alias = [
            param.name
            for param in self.params
            if param.name in aliases and param.name != param.alias
        ]
        if both_name_and_alias:
            raise ConfigurationException(
                f"Aliases {', '.join(both_name_and_alias)} could not be used "
                "as parameter name."
            )

    def get_details(self) -> dict:
        """Return a dictionary with all relevant information of a Command."""
        output = {
            "command_strings": self.command_strings,
            "user_params": [param.get_details() for param in self.params],
        }
        return output

    def __eq__(self, other: object) -> bool:
        """Overload operator ==."""
        if not isinstance(other, Command):
            return False

        return (
            self.command_strings == other.command_strings
            and self.params == other.params
        )


def resolve_all_parameters(
    str_val: str,
    param_resolver: Callable[[str, str, list[tuple[str | None, Param]]], str],
    command_name: str | None = None,
    params_values: list[tuple[str | None, Param]] | None = None,
) -> str:
    """Resolve all parameters in the string."""
    if not str_val:
        return str_val

    param_pattern: Final[Pattern] = re.compile(r"{(?P<param_name>[\w.:]+)}")
    while param_pattern.findall(str_val):
        str_val = param_pattern.sub(
            lambda m: param_resolver(
                m["param_name"], command_name or "", params_values or []
            ),
            str_val,
        )
    return str_val


def load_application_configs(
    config: Any,
    config_type: type[Any],
    is_system_required: bool = True,
) -> Any:
    """Get one config for each system supported by the application.

    The configuration could contain different parameters/commands for different
    supported systems. For each supported system this function will return separate
    config with appropriate configuration.
    """
    merged_configs = []
    supported_systems: list[NamedExecutionConfig] | None = config.get(
        "supported_systems"
    )
    if not supported_systems:
        if is_system_required:
            raise ConfigurationException("No supported systems definition provided")
        # Create an empty system to be used in the parsing below
        supported_systems = [cast(NamedExecutionConfig, {})]

    default_user_params = config.get("user_params", {})

    def merge_config(system: NamedExecutionConfig) -> Any:
        system_name = system.get("name")
        if not system_name and is_system_required:
            raise ConfigurationException(
                "Unable to read supported system definition, name is missed"
            )

        merged_config = config_type(**config)
        merged_config["supported_systems"] = [system_name] if system_name else []
        # merge default configuration and specific to the system
        merged_config["commands"] = {
            **config.get("commands", {}),
            **system.get("commands", {}),
        }

        params = {}
        tool_user_params = system.get("user_params", {})
        command_names = tool_user_params.keys() | default_user_params.keys()
        for command_name in command_names:
            if command_name not in merged_config["commands"]:
                continue

            params_default = default_user_params.get(command_name, [])
            params_tool = tool_user_params.get(command_name, [])
            if not params_default or not params_tool:
                params[command_name] = params_tool or params_default
            if params_default and params_tool:
                if any(not p.get("alias") for p in params_default):
                    raise ConfigurationException(
                        f"Default parameters for command {command_name} "
                        "should have aliases"
                    )
                if any(not p.get("alias") for p in params_tool):
                    raise ConfigurationException(
                        f"{system_name} parameters for command {command_name} "
                        "should have aliases."
                    )

                merged_by_alias = {
                    **{p.get("alias"): p for p in params_default},
                    **{p.get("alias"): p for p in params_tool},
                }
                params[command_name] = list(merged_by_alias.values())

        merged_config["user_params"] = params
        merged_config["variables"] = {
            **config.get("variables", {}),
            **system.get("variables", {}),
        }
        return merged_config

    merged_configs = [merge_config(system) for system in supported_systems]

    return merged_configs
