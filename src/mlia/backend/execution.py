# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Application execution module."""
from __future__ import annotations

import logging
import re
from typing import cast

from mlia.backend.application import Application
from mlia.backend.application import get_application
from mlia.backend.common import Backend
from mlia.backend.common import ConfigurationException
from mlia.backend.common import Param
from mlia.backend.system import get_system
from mlia.backend.system import System

logger = logging.getLogger(__name__)


class AnotherInstanceIsRunningException(Exception):
    """Concurrent execution error."""


class ExecutionContext:  # pylint: disable=too-few-public-methods
    """Command execution context."""

    def __init__(
        self,
        app: Application,
        app_params: list[str],
        system: System,
        system_params: list[str],
    ):
        """Init execution context."""
        self.app = app
        self.app_params = app_params
        self.system = system
        self.system_params = system_params

        self.param_resolver = ParamResolver(self)

        self.stdout: bytearray | None = None
        self.stderr: bytearray | None = None


class ParamResolver:
    """Parameter resolver."""

    def __init__(self, context: ExecutionContext):
        """Init parameter resolver."""
        self.ctx = context

    @staticmethod
    def resolve_user_params(
        cmd_name: str | None,
        index_or_alias: str,
        resolved_params: list[tuple[str | None, Param]] | None,
    ) -> str:
        """Resolve user params."""
        if not cmd_name or resolved_params is None:
            raise ConfigurationException("Unable to resolve user params")

        param_value: str | None = None
        param: Param | None = None

        if index_or_alias.isnumeric():
            i = int(index_or_alias)
            if i not in range(len(resolved_params)):
                raise ConfigurationException(
                    f"Invalid index {i} for user params of command {cmd_name}"
                )
            param_value, param = resolved_params[i]
        else:
            for val, par in resolved_params:
                if par.alias == index_or_alias:
                    param_value, param = val, par
                    break

            if param is None:
                raise ConfigurationException(
                    f"No user parameter for command '{cmd_name}' with "
                    f"alias '{index_or_alias}'."
                )

        if param_value:
            # We need to handle to cases of parameters here:
            # 1) Optional parameters (non-positional with a name and value)
            # 2) Positional parameters (value only, no name needed)
            # Default to empty strings for positional arguments
            param_name = ""
            separator = ""
            if param.name is not None:
                # A valid param name means we have an optional/non-positional argument:
                # The separator is an empty string in case the param_name
                # has an equal sign as we have to honour it.
                # If the parameter doesn't end with an equal sign then a
                # space character is injected to split the parameter name
                # and its value
                param_name = param.name
                separator = "" if param.name.endswith("=") else " "

            return f"{param_name}{separator}{param_value}"

        if param.name is None:
            raise ConfigurationException(
                f"Missing user parameter with alias '{index_or_alias}' for "
                f"command '{cmd_name}'."
            )

        return param.name  # flag: just return the parameter name

    def resolve_commands_and_params(
        self, backend_type: str, cmd_name: str, return_params: bool, index_or_alias: str
    ) -> str:
        """Resolve command or command's param value."""
        if backend_type == "system":
            backend = cast(Backend, self.ctx.system)
            backend_params = self.ctx.system_params
        else:  # Application backend
            backend = cast(Backend, self.ctx.app)
            backend_params = self.ctx.app_params

        if cmd_name not in backend.commands:
            raise ConfigurationException(f"Command {cmd_name} not found")

        if return_params:
            params = backend.resolved_parameters(cmd_name, backend_params)
            if index_or_alias.isnumeric():
                i = int(index_or_alias)
                if i not in range(len(params)):
                    raise ConfigurationException(
                        f"Invalid parameter index {i} for command {cmd_name}"
                    )

                param_value = params[i][0]
            else:
                param_value = None
                for value, param in params:
                    if param.alias == index_or_alias:
                        param_value = value
                        break

            if not param_value:
                raise ConfigurationException(
                    "No value for parameter with index or "
                    f"alias {index_or_alias} of command {cmd_name}."
                )
            return param_value

        if not index_or_alias.isnumeric():
            raise ConfigurationException(f"Bad command index {index_or_alias}")

        i = int(index_or_alias)
        commands = backend.build_command(cmd_name, backend_params, self.param_resolver)
        if i not in range(len(commands)):
            raise ConfigurationException(f"Invalid index {i} for command {cmd_name}")

        return commands[i]

    def resolve_variables(self, backend_type: str, var_name: str) -> str:
        """Resolve variable value."""
        if backend_type == "system":
            backend = cast(Backend, self.ctx.system)
        else:  # Application backend
            backend = cast(Backend, self.ctx.app)

        if var_name not in backend.variables:
            raise ConfigurationException(f"Unknown variable {var_name}")

        return backend.variables[var_name]

    def param_matcher(
        self,
        param_name: str,
        cmd_name: str | None,
        resolved_params: list[tuple[str | None, Param]] | None,
    ) -> str:
        """Regexp to resolve a param from the param_name."""
        # this pattern supports parameter names like "application.commands.run:0" and
        # "system.commands.run.params:0"
        # Note: 'software' is included for backward compatibility.
        commands_and_params_match = re.match(
            r"(?P<type>application|software|system)[.]commands[.]"
            r"(?P<name>\w+)"
            r"(?P<params>[.]params|)[:]"
            r"(?P<index_or_alias>\w+)",
            param_name,
        )

        if commands_and_params_match:
            backend_type, cmd_name, return_params, index_or_alias = (
                commands_and_params_match["type"],
                commands_and_params_match["name"],
                commands_and_params_match["params"],
                commands_and_params_match["index_or_alias"],
            )
            return self.resolve_commands_and_params(
                backend_type, cmd_name, bool(return_params), index_or_alias
            )

        # Note: 'software' is included for backward compatibility.
        variables_match = re.match(
            r"(?P<type>application|software|system)[.]variables:(?P<var_name>\w+)",
            param_name,
        )
        if variables_match:
            backend_type, var_name = (
                variables_match["type"],
                variables_match["var_name"],
            )
            return self.resolve_variables(backend_type, var_name)

        user_params_match = re.match(r"user_params:(?P<index_or_alias>\w+)", param_name)
        if user_params_match:
            index_or_alias = user_params_match["index_or_alias"]
            return self.resolve_user_params(cmd_name, index_or_alias, resolved_params)

        raise ConfigurationException(f"Unable to resolve parameter {param_name}")

    def param_resolver(
        self,
        param_name: str,
        cmd_name: str | None = None,
        resolved_params: list[tuple[str | None, Param]] | None = None,
    ) -> str:
        """Resolve parameter value based on current execution context."""
        # Note: 'software.*' is included for backward compatibility.
        resolved_param = None
        if param_name in ["application.name", "software.name"]:
            resolved_param = self.ctx.app.name
        elif param_name in ["application.description", "software.description"]:
            resolved_param = self.ctx.app.description
        elif self.ctx.app.config_location and (
            param_name in ["application.config_dir", "software.config_dir"]
        ):
            resolved_param = str(self.ctx.app.config_location.absolute())
        elif self.ctx.system is not None:
            if param_name == "system.name":
                resolved_param = self.ctx.system.name
            elif param_name == "system.description":
                resolved_param = self.ctx.system.description
            elif param_name == "system.config_dir" and self.ctx.system.config_location:
                resolved_param = str(self.ctx.system.config_location.absolute())

        if not resolved_param:
            resolved_param = self.param_matcher(param_name, cmd_name, resolved_params)
        return resolved_param

    def __call__(
        self,
        param_name: str,
        cmd_name: str | None = None,
        resolved_params: list[tuple[str | None, Param]] | None = None,
    ) -> str:
        """Resolve provided parameter."""
        return self.param_resolver(param_name, cmd_name, resolved_params)


def validate_parameters(
    backend: Backend, command_names: list[str], params: list[str]
) -> None:
    """Check parameters passed to backend."""
    for param in params:
        acceptable = any(
            backend.validate_parameter(command_name, param)
            for command_name in command_names
            if command_name in backend.commands
        )

        if not acceptable:
            backend_type = "System" if isinstance(backend, System) else "Application"
            raise ValueError(
                f"{backend_type} parameter '{param}' not valid for "
                f"command '{' or '.join(command_names)}'."
            )


def get_application_by_name_and_system(
    application_name: str, system_name: str
) -> Application:
    """Get application."""
    applications = get_application(application_name, system_name)
    if not applications:
        raise ValueError(
            f"Application '{application_name}' doesn't support the "
            f"system '{system_name}'."
        )

    if len(applications) != 1:
        raise ValueError(
            f"Error during getting application {application_name} for the "
            f"system {system_name}."
        )

    return applications[0]


def get_application_and_system(
    application_name: str, system_name: str
) -> tuple[Application, System]:
    """Return application and system by provided names."""
    system = get_system(system_name)
    if not system:
        raise ValueError(f"System {system_name} is not found.")

    application = get_application_by_name_and_system(application_name, system_name)

    return application, system


def run_application(
    application_name: str,
    application_params: list[str],
    system_name: str,
    system_params: list[str],
) -> ExecutionContext:
    """Run application on the provided system."""
    application, system = get_application_and_system(application_name, system_name)
    validate_parameters(application, ["run"], application_params)
    validate_parameters(system, ["run"], system_params)

    ctx = ExecutionContext(
        app=application,
        app_params=application_params,
        system=system,
        system_params=system_params,
    )

    logger.debug("Generating commands to execute")
    commands_to_run = ctx.system.build_command(
        "run", ctx.system_params, ctx.param_resolver
    )

    for command in commands_to_run:
        logger.debug("Running: %s", command)
        exit_code, ctx.stdout, ctx.stderr = ctx.system.run(command)

        if exit_code != 0:
            logger.warning("Application exited with exit code %i", exit_code)

    return ctx
