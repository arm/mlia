# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Application execution module."""
import itertools
import json
import random
import re
import string
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import ContextManager
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypedDict

from filelock import FileLock
from filelock import Timeout

from mlia.backend.application import Application
from mlia.backend.application import get_application
from mlia.backend.common import Backend
from mlia.backend.common import ConfigurationException
from mlia.backend.common import DataPaths
from mlia.backend.common import Param
from mlia.backend.common import parse_raw_parameter
from mlia.backend.common import resolve_all_parameters
from mlia.backend.fs import recreate_directory
from mlia.backend.fs import remove_directory
from mlia.backend.fs import valid_for_filename
from mlia.backend.output_parser import Base64OutputParser
from mlia.backend.output_parser import OutputParser
from mlia.backend.output_parser import RegexOutputParser
from mlia.backend.proc import run_and_wait
from mlia.backend.system import ControlledSystem
from mlia.backend.system import get_system
from mlia.backend.system import StandaloneSystem
from mlia.backend.system import System


class AnotherInstanceIsRunningException(Exception):
    """Concurrent execution error."""


class ConnectionException(Exception):
    """Connection exception."""


class ExecutionParams(TypedDict, total=False):
    """Execution parameters."""

    disable_locking: bool
    unique_build_dir: bool


class ExecutionContext:
    """Command execution context."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        app: Application,
        app_params: List[str],
        system: Optional[System],
        system_params: List[str],
        custom_deploy_data: Optional[List[DataPaths]] = None,
        execution_params: Optional[ExecutionParams] = None,
        report_file: Optional[Path] = None,
    ):
        """Init execution context."""
        self.app = app
        self.app_params = app_params
        self.custom_deploy_data = custom_deploy_data or []
        self.system = system
        self.system_params = system_params
        self.execution_params = execution_params or ExecutionParams()
        self.report_file = report_file

        self.reporter: Optional[Reporter]
        if self.report_file:
            # Create reporter with output parsers
            parsers: List[OutputParser] = []
            if system and system.reporting:
                # Add RegexOutputParser, if it is configured in the system
                parsers.append(RegexOutputParser("system", system.reporting["regex"]))
            # Add Base64 parser for applications
            parsers.append(Base64OutputParser("application"))
            self.reporter = Reporter(parsers=parsers)
        else:
            self.reporter = None  # No reporter needed.

        self.param_resolver = ParamResolver(self)
        self._resolved_build_dir: Optional[Path] = None

        self.stdout: Optional[bytearray] = None
        self.stderr: Optional[bytearray] = None

    @property
    def is_deploy_needed(self) -> bool:
        """Check if application requires data deployment."""
        return len(self.app.get_deploy_data()) > 0 or len(self.custom_deploy_data) > 0

    @property
    def is_locking_required(self) -> bool:
        """Return true if any form of locking required."""
        return not self._disable_locking() and (
            self.app.lock or (self.system is not None and self.system.lock)
        )

    @property
    def is_build_required(self) -> bool:
        """Return true if application build required."""
        return "build" in self.app.commands

    @property
    def is_unique_build_dir_required(self) -> bool:
        """Return true if unique build dir required."""
        return self.execution_params.get("unique_build_dir", False)

    def build_dir(self) -> Path:
        """Return resolved application build dir."""
        if self._resolved_build_dir is not None:
            return self._resolved_build_dir

        if (
            not isinstance(self.app.config_location, Path)
            or not self.app.config_location.is_dir()
        ):
            raise ConfigurationException(
                "Application {} has wrong config location".format(self.app.name)
            )

        _build_dir = self.app.build_dir
        if _build_dir:
            _build_dir = resolve_all_parameters(_build_dir, self.param_resolver)

        if not _build_dir:
            raise ConfigurationException(
                "No build directory defined for the app {}".format(self.app.name)
            )

        if self.is_unique_build_dir_required:
            random_suffix = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=7)
            )
            _build_dir = "{}_{}".format(_build_dir, random_suffix)

        self._resolved_build_dir = self.app.config_location / _build_dir
        return self._resolved_build_dir

    def _disable_locking(self) -> bool:
        """Return true if locking should be disabled."""
        return self.execution_params.get("disable_locking", False)


class ParamResolver:
    """Parameter resolver."""

    def __init__(self, context: ExecutionContext):
        """Init parameter resolver."""
        self.ctx = context

    @staticmethod
    def resolve_user_params(
        cmd_name: Optional[str],
        index_or_alias: str,
        resolved_params: Optional[List[Tuple[Optional[str], Param]]],
    ) -> str:
        """Resolve user params."""
        if not cmd_name or resolved_params is None:
            raise ConfigurationException("Unable to resolve user params")

        param_value: Optional[str] = None
        param: Optional[Param] = None

        if index_or_alias.isnumeric():
            i = int(index_or_alias)
            if i not in range(len(resolved_params)):
                raise ConfigurationException(
                    "Invalid index {} for user params of command {}".format(i, cmd_name)
                )
            param_value, param = resolved_params[i]
        else:
            for val, par in resolved_params:
                if par.alias == index_or_alias:
                    param_value, param = val, par
                    break

            if param is None:
                raise ConfigurationException(
                    "No user parameter for command '{}' with alias '{}'.".format(
                        cmd_name, index_or_alias
                    )
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

            return "{param_name}{separator}{param_value}".format(
                param_name=param_name,
                separator=separator,
                param_value=param_value,
            )

        if param.name is None:
            raise ConfigurationException(
                "Missing user parameter with alias '{}' for command '{}'.".format(
                    index_or_alias, cmd_name
                )
            )

        return param.name  # flag: just return the parameter name

    def resolve_commands_and_params(
        self, backend_type: str, cmd_name: str, return_params: bool, index_or_alias: str
    ) -> str:
        """Resolve command or command's param value."""
        if backend_type == "system":
            backend = cast(Backend, self.ctx.system)
            backend_params = self.ctx.system_params
        else:  # Application or Tool backend
            backend = cast(Backend, self.ctx.app)
            backend_params = self.ctx.app_params

        if cmd_name not in backend.commands:
            raise ConfigurationException("Command {} not found".format(cmd_name))

        if return_params:
            params = backend.resolved_parameters(cmd_name, backend_params)
            if index_or_alias.isnumeric():
                i = int(index_or_alias)
                if i not in range(len(params)):
                    raise ConfigurationException(
                        "Invalid parameter index {} for command {}".format(i, cmd_name)
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
                    (
                        "No value for parameter with index or alias {} of command {}"
                    ).format(index_or_alias, cmd_name)
                )
            return param_value

        if not index_or_alias.isnumeric():
            raise ConfigurationException("Bad command index {}".format(index_or_alias))

        i = int(index_or_alias)
        commands = backend.build_command(cmd_name, backend_params, self.param_resolver)
        if i not in range(len(commands)):
            raise ConfigurationException(
                "Invalid index {} for command {}".format(i, cmd_name)
            )

        return commands[i]

    def resolve_variables(self, backend_type: str, var_name: str) -> str:
        """Resolve variable value."""
        if backend_type == "system":
            backend = cast(Backend, self.ctx.system)
        else:  # Application or Tool backend
            backend = cast(Backend, self.ctx.app)

        if var_name not in backend.variables:
            raise ConfigurationException("Unknown variable {}".format(var_name))

        return backend.variables[var_name]

    def param_matcher(
        self,
        param_name: str,
        cmd_name: Optional[str],
        resolved_params: Optional[List[Tuple[Optional[str], Param]]],
    ) -> str:
        """Regexp to resolve a param from the param_name."""
        # this pattern supports parameter names like "application.commands.run:0" and
        # "system.commands.run.params:0"
        # Note: 'software' is included for backward compatibility.
        commands_and_params_match = re.match(
            r"(?P<type>application|software|tool|system)[.]commands[.]"
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
            r"(?P<type>application|software|tool|system)[.]variables:(?P<var_name>\w+)",
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

        raise ConfigurationException(
            "Unable to resolve parameter {}".format(param_name)
        )

    def param_resolver(
        self,
        param_name: str,
        cmd_name: Optional[str] = None,
        resolved_params: Optional[List[Tuple[Optional[str], Param]]] = None,
    ) -> str:
        """Resolve parameter value based on current execution context."""
        # Note: 'software.*' is included for backward compatibility.
        resolved_param = None
        if param_name in ["application.name", "tool.name", "software.name"]:
            resolved_param = self.ctx.app.name
        elif param_name in [
            "application.description",
            "tool.description",
            "software.description",
        ]:
            resolved_param = self.ctx.app.description
        elif self.ctx.app.config_location and (
            param_name
            in ["application.config_dir", "tool.config_dir", "software.config_dir"]
        ):
            resolved_param = str(self.ctx.app.config_location.absolute())
        elif self.ctx.app.build_dir and (
            param_name
            in ["application.build_dir", "tool.build_dir", "software.build_dir"]
        ):
            resolved_param = str(self.ctx.build_dir().absolute())
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
        cmd_name: Optional[str] = None,
        resolved_params: Optional[List[Tuple[Optional[str], Param]]] = None,
    ) -> str:
        """Resolve provided parameter."""
        return self.param_resolver(param_name, cmd_name, resolved_params)


class Reporter:
    """Report metrics from the simulation output."""

    def __init__(self, parsers: Optional[List[OutputParser]] = None) -> None:
        """Create an empty reporter (i.e. no parsers registered)."""
        self.parsers: List[OutputParser] = parsers if parsers is not None else []
        self._report: Dict[str, Any] = defaultdict(lambda: defaultdict(dict))

    def parse(self, output: bytearray) -> None:
        """Parse output and append parsed metrics to internal report dict."""
        for parser in self.parsers:
            # Merge metrics from different parsers (do not overwrite)
            self._report[parser.name]["metrics"].update(parser(output))

    def get_filtered_output(self, output: bytearray) -> bytearray:
        """Filter the output according to each parser."""
        for parser in self.parsers:
            output = parser.filter_out_parsed_content(output)
        return output

    def report(self, ctx: ExecutionContext) -> Dict[str, Any]:
        """Add static simulation info to parsed data and return the report."""
        report: Dict[str, Any] = defaultdict(dict)
        # Add static simulation info
        report.update(self._static_info(ctx))
        # Add metrics parsed from the output
        for key, val in self._report.items():
            report[key].update(val)
        return report

    @staticmethod
    def save(report: Dict[str, Any], report_file: Path) -> None:
        """Save the report to a JSON file."""
        with open(report_file, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

    @staticmethod
    def _compute_all_params(cli_params: List[str], backend: Backend) -> Dict[str, str]:
        """
        Build a dict of all parameters, {name:value}.

        Param values taken from command line if specified, defaults otherwise.
        """
        # map of params passed from the cli ["p1=v1","p2=v2"] -> {"p1":"v1", "p2":"v2"}
        app_params_map = dict(parse_raw_parameter(expr) for expr in cli_params)

        # a map of params declared in the application, with values taken from the CLI,
        # defaults otherwise
        all_params = {
            (p.alias or p.name): app_params_map.get(
                cast(str, p.name), cast(str, p.default_value)
            )
            for cmd in backend.commands.values()
            for p in cmd.params
        }
        return cast(Dict[str, str], all_params)

    @staticmethod
    def _static_info(ctx: ExecutionContext) -> Dict[str, Any]:
        """Extract static simulation information from the context."""
        if ctx.system is None:
            raise ValueError("No system available to report.")

        info = {
            "system": {
                "name": ctx.system.name,
                "params": Reporter._compute_all_params(ctx.system_params, ctx.system),
            },
            "application": {
                "name": ctx.app.name,
                "params": Reporter._compute_all_params(ctx.app_params, ctx.app),
            },
        }
        return info


def validate_parameters(
    backend: Backend, command_names: List[str], params: List[str]
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
                "{} parameter '{}' not valid for command '{}'".format(
                    backend_type, param, " or ".join(command_names)
                )
            )


def get_application_by_name_and_system(
    application_name: str, system_name: str
) -> Application:
    """Get application."""
    applications = get_application(application_name, system_name)
    if not applications:
        raise ValueError(
            "Application '{}' doesn't support the system '{}'".format(
                application_name, system_name
            )
        )

    if len(applications) != 1:
        raise ValueError(
            "Error during getting application {} for the system {}".format(
                application_name, system_name
            )
        )

    return applications[0]


def get_application_and_system(
    application_name: str, system_name: str
) -> Tuple[Application, System]:
    """Return application and system by provided names."""
    system = get_system(system_name)
    if not system:
        raise ValueError("System {} is not found".format(system_name))

    application = get_application_by_name_and_system(application_name, system_name)

    return application, system


# pylint: disable=too-many-arguments
def run_application(
    application_name: str,
    application_params: List[str],
    system_name: str,
    system_params: List[str],
    custom_deploy_data: List[DataPaths],
    report_file: Optional[Path] = None,
) -> ExecutionContext:
    """Run application on the provided system."""
    application, system = get_application_and_system(application_name, system_name)
    validate_parameters(application, ["build", "run"], application_params)
    validate_parameters(system, ["build", "run"], system_params)

    execution_params = ExecutionParams()
    if isinstance(system, StandaloneSystem):
        execution_params["disable_locking"] = True
        execution_params["unique_build_dir"] = True

    ctx = ExecutionContext(
        app=application,
        app_params=application_params,
        system=system,
        system_params=system_params,
        custom_deploy_data=custom_deploy_data,
        execution_params=execution_params,
        report_file=report_file,
    )

    with build_dir_manager(ctx):
        if ctx.is_build_required:
            execute_application_command_build(ctx)

        execute_application_command_run(ctx)

    return ctx


def execute_application_command_build(ctx: ExecutionContext) -> None:
    """Execute application command 'build'."""
    with ExitStack() as context_stack:
        for manager in get_context_managers("build", ctx):
            context_stack.enter_context(manager(ctx))

        build_dir = ctx.build_dir()
        recreate_directory(build_dir)

        build_commands = ctx.app.build_command(
            "build", ctx.app_params, ctx.param_resolver
        )
        execute_commands_locally(build_commands, build_dir)


def execute_commands_locally(commands: List[str], cwd: Path) -> None:
    """Execute list of commands locally."""
    for command in commands:
        print("Running: {}".format(command))
        run_and_wait(
            command, cwd, terminate_on_error=True, out=sys.stdout, err=sys.stderr
        )


def execute_application_command_run(ctx: ExecutionContext) -> None:
    """Execute application command."""
    assert ctx.system is not None, "System must be provided."
    if ctx.is_deploy_needed and not ctx.system.supports_deploy:
        raise ConfigurationException(
            "System {} does not support data deploy".format(ctx.system.name)
        )

    with ExitStack() as context_stack:
        for manager in get_context_managers("run", ctx):
            context_stack.enter_context(manager(ctx))

        print("Generating commands to execute")
        commands_to_run = build_run_commands(ctx)

        if ctx.system.connectable:
            establish_connection(ctx)

        if ctx.system.supports_deploy:
            deploy_data(ctx)

        for command in commands_to_run:
            print("Running: {}".format(command))
            exit_code, ctx.stdout, ctx.stderr = ctx.system.run(command)

            if exit_code != 0:
                print("Application exited with exit code {}".format(exit_code))

            if ctx.reporter:
                ctx.reporter.parse(ctx.stdout)
                ctx.stdout = ctx.reporter.get_filtered_output(ctx.stdout)

        if ctx.reporter:
            report = ctx.reporter.report(ctx)
            ctx.reporter.save(report, cast(Path, ctx.report_file))


def establish_connection(
    ctx: ExecutionContext, retries: int = 90, interval: float = 15.0
) -> None:
    """Establish connection with the system."""
    assert ctx.system is not None, "System is required."
    host, port = ctx.system.connection_details()
    print(
        "Trying to establish connection with '{}:{}' - "
        "{} retries every {} seconds ".format(host, port, retries, interval),
        end="",
    )

    try:
        for _ in range(retries):
            print(".", end="", flush=True)

            if ctx.system.establish_connection():
                break

            if isinstance(ctx.system, ControlledSystem) and not ctx.system.is_running():
                print(
                    "\n\n---------- {} execution failed ----------".format(
                        ctx.system.name
                    )
                )
                stdout, stderr = ctx.system.get_output()
                print(stdout)
                print(stderr)

                raise Exception("System is not running")

            wait(interval)
        else:
            raise ConnectionException("Couldn't connect to '{}:{}'.".format(host, port))
    finally:
        print()


def wait(interval: float) -> None:
    """Wait for a period of time."""
    time.sleep(interval)


def deploy_data(ctx: ExecutionContext) -> None:
    """Deploy data to the system."""
    assert ctx.system is not None, "System is required."
    for item in itertools.chain(ctx.app.get_deploy_data(), ctx.custom_deploy_data):
        print("Deploying {} onto {}".format(item.src, item.dst))
        ctx.system.deploy(item.src, item.dst)


def build_run_commands(ctx: ExecutionContext) -> List[str]:
    """Build commands to run application."""
    if isinstance(ctx.system, StandaloneSystem):
        return ctx.system.build_command("run", ctx.system_params, ctx.param_resolver)

    return ctx.app.build_command("run", ctx.app_params, ctx.param_resolver)


@contextmanager
def controlled_system_manager(ctx: ExecutionContext) -> Generator[None, None, None]:
    """Context manager used for system initialisation before run."""
    system = cast(ControlledSystem, ctx.system)
    commands = system.build_command("run", ctx.system_params, ctx.param_resolver)
    pid_file_path: Optional[Path] = None
    if ctx.is_locking_required:
        file_lock_path = get_file_lock_path(ctx)
        pid_file_path = file_lock_path.parent / "{}.pid".format(file_lock_path.stem)

    system.start(commands, ctx.is_locking_required, pid_file_path)
    try:
        yield
    finally:
        print("Shutting down sequence...")
        print("Stopping {}... (It could take few seconds)".format(system.name))
        system.stop(wait=True)
        print("{} stopped successfully.".format(system.name))


@contextmanager
def lock_execution_manager(ctx: ExecutionContext) -> Generator[None, None, None]:
    """Lock execution manager."""
    file_lock_path = get_file_lock_path(ctx)
    file_lock = FileLock(str(file_lock_path))

    try:
        file_lock.acquire(timeout=1)
    except Timeout as error:
        raise AnotherInstanceIsRunningException() from error

    try:
        yield
    finally:
        file_lock.release()


def get_file_lock_path(ctx: ExecutionContext, lock_dir: Path = Path("/tmp")) -> Path:
    """Get file lock path."""
    lock_modules = []
    if ctx.app.lock:
        lock_modules.append(ctx.app.name)
    if ctx.system is not None and ctx.system.lock:
        lock_modules.append(ctx.system.name)
    lock_filename = ""
    if lock_modules:
        lock_filename = "_".join(["middleware"] + lock_modules) + ".lock"

    if lock_filename:
        lock_filename = resolve_all_parameters(lock_filename, ctx.param_resolver)
        lock_filename = valid_for_filename(lock_filename)

    if not lock_filename:
        raise ConfigurationException("No filename for lock provided")

    if not isinstance(lock_dir, Path) or not lock_dir.is_dir():
        raise ConfigurationException(
            "Invalid directory {} for lock files provided".format(lock_dir)
        )

    return lock_dir / lock_filename


@contextmanager
def build_dir_manager(ctx: ExecutionContext) -> Generator[None, None, None]:
    """Build directory manager."""
    try:
        yield
    finally:
        if (
            ctx.is_build_required
            and ctx.is_unique_build_dir_required
            and ctx.build_dir().is_dir()
        ):
            remove_directory(ctx.build_dir())


def get_context_managers(
    command_name: str, ctx: ExecutionContext
) -> Sequence[Callable[[ExecutionContext], ContextManager[None]]]:
    """Get context manager for the system."""
    managers = []

    if ctx.is_locking_required:
        managers.append(lock_execution_manager)

    if command_name == "run":
        if isinstance(ctx.system, ControlledSystem):
            managers.append(controlled_system_manager)

    return managers
