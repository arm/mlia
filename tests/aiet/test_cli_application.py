# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=attribute-defined-outside-init,no-member,line-too-long,too-many-arguments,too-many-locals,redefined-outer-name,too-many-lines
"""Module for testing CLI application subcommand."""
import base64
import json
import re
import time
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from typing import Generator
from typing import IO
from typing import List
from typing import Optional
from typing import TypedDict
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
from filelock import FileLock

from aiet.backend.application import Application
from aiet.backend.config import ApplicationConfig
from aiet.backend.config import LocalProtocolConfig
from aiet.backend.config import SSHConfig
from aiet.backend.config import SystemConfig
from aiet.backend.config import UserParamConfig
from aiet.backend.output_parser import Base64OutputParser
from aiet.backend.protocol import SSHProtocol
from aiet.backend.system import load_system
from aiet.cli.application import application_cmd
from aiet.cli.application import details_cmd
from aiet.cli.application import execute_cmd
from aiet.cli.application import install_cmd
from aiet.cli.application import list_cmd
from aiet.cli.application import parse_payload_run_config
from aiet.cli.application import remove_cmd
from aiet.cli.application import run_cmd
from aiet.cli.common import MiddlewareExitCode


def test_application_cmd() -> None:
    """Test application commands."""
    commands = ["list", "details", "install", "remove", "execute", "run"]
    assert all(command in application_cmd.commands for command in commands)


@pytest.mark.parametrize("format_", ["json", "cli"])
def test_application_cmd_context(cli_runner: CliRunner, format_: str) -> None:
    """Test setting command context parameters."""
    result = cli_runner.invoke(application_cmd, ["--format", format_])
    # command should fail if no subcommand provided
    assert result.exit_code == 2

    result = cli_runner.invoke(application_cmd, ["--format", format_, "list"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "format_, system_name, expected_output",
    [
        (
            "json",
            None,
            '{"type": "application", "available": ["application_1", "application_2"]}\n',
        ),
        (
            "json",
            "system_1",
            '{"type": "application", "available": ["application_1"]}\n',
        ),
        ("cli", None, "Available applications:\n\napplication_1\napplication_2\n"),
        ("cli", "system_1", "Available applications:\n\napplication_1\n"),
    ],
)
def test_list_cmd(
    cli_runner: CliRunner,
    monkeypatch: Any,
    format_: str,
    system_name: str,
    expected_output: str,
) -> None:
    """Test available applications commands."""
    # Mock some applications
    mock_application_1 = MagicMock(spec=Application)
    mock_application_1.name = "application_1"
    mock_application_1.can_run_on.return_value = system_name == "system_1"
    mock_application_2 = MagicMock(spec=Application)
    mock_application_2.name = "application_2"
    mock_application_2.can_run_on.return_value = system_name == "system_2"

    # Monkey patch the call get_available_applications
    mock_available_applications = MagicMock()
    mock_available_applications.return_value = [mock_application_1, mock_application_2]

    monkeypatch.setattr(
        "aiet.backend.application.get_available_applications",
        mock_available_applications,
    )

    obj = {"format": format_}
    args = []
    if system_name:
        list_cmd.params[0].type = click.Choice([system_name])
        args = ["--system", system_name]
    result = cli_runner.invoke(list_cmd, obj=obj, args=args)
    assert result.output == expected_output


def get_test_application() -> Application:
    """Return test system details."""
    config = ApplicationConfig(
        name="application",
        description="test",
        build_dir="",
        supported_systems=[],
        deploy_data=[],
        user_params={},
        commands={
            "clean": ["clean"],
            "build": ["build"],
            "run": ["run"],
            "post_run": ["post_run"],
        },
    )

    return Application(config)


def get_details_cmd_json_output() -> str:
    """Get JSON output for details command."""
    json_output = """
[
    {
        "type": "application",
        "name": "application",
        "description": "test",
        "supported_systems": [],
        "commands": {
            "clean": {
                "command_strings": [
                    "clean"
                ],
                "user_params": []
            },
            "build": {
                "command_strings": [
                    "build"
                ],
                "user_params": []
            },
            "run": {
                "command_strings": [
                    "run"
                ],
                "user_params": []
            },
            "post_run": {
                "command_strings": [
                    "post_run"
                ],
                "user_params": []
            }
        }
    }
]"""
    return json.dumps(json.loads(json_output)) + "\n"


def get_details_cmd_console_output() -> str:
    """Get console output for details command."""
    return (
        'Application "application" details'
        + "\nDescription: test"
        + "\n\nSupported systems: "
        + "\n\nclean commands:"
        + "\nCommands: ['clean']"
        + "\n\nbuild commands:"
        + "\nCommands: ['build']"
        + "\n\nrun commands:"
        + "\nCommands: ['run']"
        + "\n\npost_run commands:"
        + "\nCommands: ['post_run']"
        + "\n"
    )


@pytest.mark.parametrize(
    "application_name,format_, expected_output",
    [
        ("application", "json", get_details_cmd_json_output()),
        ("application", "cli", get_details_cmd_console_output()),
    ],
)
def test_details_cmd(
    cli_runner: CliRunner,
    monkeypatch: Any,
    application_name: str,
    format_: str,
    expected_output: str,
) -> None:
    """Test application details command."""
    monkeypatch.setattr(
        "aiet.cli.application.get_application",
        MagicMock(return_value=[get_test_application()]),
    )

    details_cmd.params[0].type = click.Choice(["application"])
    result = cli_runner.invoke(
        details_cmd, obj={"format": format_}, args=["--name", application_name]
    )
    assert result.exception is None
    assert result.output == expected_output


def test_details_cmd_wrong_system(cli_runner: CliRunner, monkeypatch: Any) -> None:
    """Test details command fails if application is not supported by the system."""
    monkeypatch.setattr(
        "aiet.backend.execution.get_application", MagicMock(return_value=[])
    )

    details_cmd.params[0].type = click.Choice(["application"])
    details_cmd.params[1].type = click.Choice(["system"])
    result = cli_runner.invoke(
        details_cmd, args=["--name", "application", "--system", "system"]
    )
    assert result.exit_code == 2
    assert (
        "Application 'application' doesn't support the system 'system'" in result.stdout
    )


def test_install_cmd(cli_runner: CliRunner, monkeypatch: Any) -> None:
    """Test install application command."""
    mock_install_application = MagicMock()
    monkeypatch.setattr(
        "aiet.cli.application.install_application", mock_install_application
    )

    args = ["--source", "test"]
    cli_runner.invoke(install_cmd, args=args)
    mock_install_application.assert_called_once_with(Path("test"))


def test_remove_cmd(cli_runner: CliRunner, monkeypatch: Any) -> None:
    """Test remove application command."""
    mock_remove_application = MagicMock()
    monkeypatch.setattr(
        "aiet.cli.application.remove_application", mock_remove_application
    )
    remove_cmd.params[0].type = click.Choice(["test"])

    args = ["--directory_name", "test"]
    cli_runner.invoke(remove_cmd, args=args)
    mock_remove_application.assert_called_once_with("test")


class ExecutionCase(TypedDict, total=False):
    """Execution case."""

    args: List[str]
    lock_path: str
    can_establish_connection: bool
    establish_connection_delay: int
    app_exit_code: int
    exit_code: int
    output: str


@pytest.mark.parametrize(
    "application_config, system_config, executions",
    [
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                config_location=Path("wrong_location"),
                commands={"build": ["echo build {application.name}"]},
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                config_location=Path("wrong_location"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "build"],
                    exit_code=MiddlewareExitCode.CONFIGURATION_ERROR,
                    output="Error: Application test_application has wrong config location\n",
                )
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                build_dir="build",
                deploy_data=[("sample_file", "/tmp/sample_file")],
                commands={"build": ["echo build {application.name}"]},
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "run"],
                    exit_code=MiddlewareExitCode.CONFIGURATION_ERROR,
                    output="Error: System test_system does not support data deploy\n",
                )
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                commands={"build": ["echo build {application.name}"]},
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "build"],
                    exit_code=MiddlewareExitCode.CONFIGURATION_ERROR,
                    output="Error: No build directory defined for the app test_application\n",
                )
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["new_system"],
                build_dir="build",
                commands={
                    "build": ["echo build {application.name} with {user_params:0}"]
                },
                user_params={
                    "build": [
                        UserParamConfig(
                            name="param",
                            description="sample parameter",
                            default_value="default",
                            values=["val1", "val2", "val3"],
                        )
                    ]
                },
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "build"],
                    exit_code=1,
                    output="Error: Application 'test_application' doesn't support the system 'test_system'\n",
                )
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                build_dir="build",
                commands={"build": ["false"]},
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "build"],
                    exit_code=MiddlewareExitCode.BACKEND_ERROR,
                    output="""Running: false
Error: Execution failed. Please check output for the details.\n""",
                )
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                lock=True,
                build_dir="build",
                commands={
                    "build": ["echo build {application.name} with {user_params:0}"]
                },
                user_params={
                    "build": [
                        UserParamConfig(
                            name="param",
                            description="sample parameter",
                            default_value="default",
                            values=["val1", "val2", "val3"],
                        )
                    ]
                },
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                lock=True,
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "build"],
                    exit_code=MiddlewareExitCode.SUCCESS,
                    output="""Running: echo build test_application with param default
build test_application with param default\n""",
                ),
                ExecutionCase(
                    args=["-c", "build"],
                    lock_path="/tmp/middleware_test_application_test_system.lock",
                    exit_code=MiddlewareExitCode.CONCURRENT_ERROR,
                    output="Error: Another instance of the system is running\n",
                ),
                ExecutionCase(
                    args=["-c", "build", "--param=param=val3"],
                    exit_code=MiddlewareExitCode.SUCCESS,
                    output="""Running: echo build test_application with param val3
build test_application with param val3\n""",
                ),
                ExecutionCase(
                    args=["-c", "build", "--param=param=newval"],
                    exit_code=1,
                    output="Error: Application parameter 'param=newval' not valid for command 'build'\n",
                ),
                ExecutionCase(
                    args=["-c", "some_command"],
                    exit_code=MiddlewareExitCode.CONFIGURATION_ERROR,
                    output="Error: Unsupported command some_command\n",
                ),
                ExecutionCase(
                    args=["-c", "run"],
                    exit_code=MiddlewareExitCode.SUCCESS,
                    output="""Generating commands to execute
Running: echo run test_application on test_system
run test_application on test_system\n""",
                ),
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                deploy_data=[("sample_file", "/tmp/sample_file")],
                commands={
                    "run": [
                        "echo run {application.name} with {user_params:param} on {system.name}"
                    ]
                },
                user_params={
                    "run": [
                        UserParamConfig(
                            name="param=",
                            description="sample parameter",
                            default_value="default",
                            values=["val1", "val2", "val3"],
                            alias="param",
                        )
                    ]
                },
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                lock=True,
                data_transfer=SSHConfig(
                    protocol="ssh",
                    username="username",
                    password="password",
                    hostname="localhost",
                    port="8022",
                ),
                commands={"run": ["sleep 100"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "run"],
                    exit_code=MiddlewareExitCode.SUCCESS,
                    output="""Generating commands to execute
Trying to establish connection with 'localhost:8022' - 90 retries every 15.0 seconds .
Deploying {application.config_location}/sample_file onto /tmp/sample_file
Running: echo run test_application with param=default on test_system
Shutting down sequence...
Stopping test_system... (It could take few seconds)
test_system stopped successfully.\n""",
                ),
                ExecutionCase(
                    args=["-c", "run"],
                    lock_path="/tmp/middleware_test_system.lock",
                    exit_code=MiddlewareExitCode.CONCURRENT_ERROR,
                    output="Error: Another instance of the system is running\n",
                ),
                ExecutionCase(
                    args=[
                        "-c",
                        "run",
                        "--deploy={application.config_location}/sample_file:/tmp/sample_file",
                    ],
                    exit_code=0,
                    output="""Generating commands to execute
Trying to establish connection with 'localhost:8022' - 90 retries every 15.0 seconds .
Deploying {application.config_location}/sample_file onto /tmp/sample_file
Deploying {application.config_location}/sample_file onto /tmp/sample_file
Running: echo run test_application with param=default on test_system
Shutting down sequence...
Stopping test_system... (It could take few seconds)
test_system stopped successfully.\n""",
                ),
                ExecutionCase(
                    args=["-c", "run"],
                    app_exit_code=1,
                    exit_code=0,
                    output="""Generating commands to execute
Trying to establish connection with 'localhost:8022' - 90 retries every 15.0 seconds .
Deploying {application.config_location}/sample_file onto /tmp/sample_file
Running: echo run test_application with param=default on test_system
Application exited with exit code 1
Shutting down sequence...
Stopping test_system... (It could take few seconds)
test_system stopped successfully.\n""",
                ),
                ExecutionCase(
                    args=["-c", "run"],
                    exit_code=MiddlewareExitCode.CONNECTION_ERROR,
                    can_establish_connection=False,
                    output="""Generating commands to execute
Trying to establish connection with 'localhost:8022' - 90 retries every 15.0 seconds ..........................................................................................
Shutting down sequence...
Stopping test_system... (It could take few seconds)
test_system stopped successfully.
Error: Couldn't connect to 'localhost:8022'.\n""",
                ),
                ExecutionCase(
                    args=["-c", "run", "--deploy=bad_format"],
                    exit_code=1,
                    output="Error: Invalid deploy parameter 'bad_format' for command run\n",
                ),
                ExecutionCase(
                    args=["-c", "run", "--deploy=:"],
                    exit_code=1,
                    output="Error: Invalid deploy parameter ':' for command run\n",
                ),
                ExecutionCase(
                    args=["-c", "run", "--deploy=   :   "],
                    exit_code=1,
                    output="Error: Invalid deploy parameter '   :   ' for command run\n",
                ),
                ExecutionCase(
                    args=["-c", "run", "--deploy=some_src_file:"],
                    exit_code=1,
                    output="Error: Invalid deploy parameter 'some_src_file:' for command run\n",
                ),
                ExecutionCase(
                    args=["-c", "run", "--deploy=:some_dst_file"],
                    exit_code=1,
                    output="Error: Invalid deploy parameter ':some_dst_file' for command run\n",
                ),
                ExecutionCase(
                    args=["-c", "run", "--deploy=unknown_file:/tmp/dest"],
                    exit_code=1,
                    output="Error: Path unknown_file does not exist\n",
                ),
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                commands={
                    "run": [
                        "echo run {application.name} with {user_params:param} on {system.name}"
                    ]
                },
                user_params={
                    "run": [
                        UserParamConfig(
                            name="param=",
                            description="sample parameter",
                            default_value="default",
                            values=["val1", "val2", "val3"],
                            alias="param",
                        )
                    ]
                },
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=SSHConfig(
                    protocol="ssh",
                    username="username",
                    password="password",
                    hostname="localhost",
                    port="8022",
                ),
                commands={"run": ["echo Unable to start system"]},
            ),
            [
                ExecutionCase(
                    args=["-c", "run"],
                    exit_code=4,
                    can_establish_connection=False,
                    establish_connection_delay=1,
                    output="""Generating commands to execute
Trying to establish connection with 'localhost:8022' - 90 retries every 15.0 seconds .

---------- test_system execution failed ----------
Unable to start system



Shutting down sequence...
Stopping test_system... (It could take few seconds)
test_system stopped successfully.
Error: Execution failed. Please check output for the details.\n""",
                )
            ],
        ],
    ],
)
def test_application_command_execution(
    application_config: ApplicationConfig,
    system_config: SystemConfig,
    executions: List[ExecutionCase],
    tmpdir: Any,
    cli_runner: CliRunner,
    monkeypatch: Any,
) -> None:
    """Test application command execution."""

    @contextmanager
    def lock_execution(lock_path: str) -> Generator[None, None, None]:
        lock = FileLock(lock_path)
        lock.acquire(timeout=1)

        try:
            yield
        finally:
            lock.release()

    def replace_vars(str_val: str) -> str:
        """Replace variables."""
        application_config_location = str(
            application_config["config_location"].absolute()
        )

        return str_val.replace(
            "{application.config_location}", application_config_location
        )

    for execution in executions:
        init_execution_test(
            monkeypatch,
            tmpdir,
            application_config,
            system_config,
            can_establish_connection=execution.get("can_establish_connection", True),
            establish_conection_delay=execution.get("establish_connection_delay", 0),
            remote_app_exit_code=execution.get("app_exit_code", 0),
        )

        lock_path = execution.get("lock_path")

        with ExitStack() as stack:
            if lock_path:
                stack.enter_context(lock_execution(lock_path))

            args = [replace_vars(arg) for arg in execution["args"]]

            result = cli_runner.invoke(
                execute_cmd,
                args=["-n", application_config["name"], "-s", system_config["name"]]
                + args,
            )
            output = replace_vars(execution["output"])
            assert result.exit_code == execution["exit_code"]
            assert result.stdout == output


@pytest.fixture(params=[False, True], ids=["run-cli", "run-json"])
def payload_path_or_none(request: Any, tmp_path_factory: Any) -> Optional[Path]:
    """Drives tests for run command so that it executes them both to use a json file, and to use CLI."""
    if request.param:
        ret: Path = tmp_path_factory.getbasetemp() / "system_config_payload_file.json"
        return ret
    return None


def write_system_payload_config(
    payload_file: IO[str],
    application_config: ApplicationConfig,
    system_config: SystemConfig,
) -> None:
    """Write a json payload file for the given test configuration."""
    payload_dict = {
        "id": system_config["name"],
        "arguments": {
            "application": application_config["name"],
        },
    }
    json.dump(payload_dict, payload_file)


@pytest.mark.parametrize(
    "application_config, system_config, executions",
    [
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                build_dir="build",
                commands={
                    "build": ["echo build {application.name} with {user_params:0}"]
                },
                user_params={
                    "build": [
                        UserParamConfig(
                            name="param",
                            description="sample parameter",
                            default_value="default",
                            values=["val1", "val2", "val3"],
                        )
                    ]
                },
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={"run": ["echo run {application.name} on {system.name}"]},
            ),
            [
                ExecutionCase(
                    args=[],
                    exit_code=MiddlewareExitCode.SUCCESS,
                    output="""Running: echo build test_application with param default
build test_application with param default
Generating commands to execute
Running: echo run test_application on test_system
run test_application on test_system\n""",
                )
            ],
        ],
        [
            ApplicationConfig(
                name="test_application",
                description="Test application",
                supported_systems=["test_system"],
                commands={
                    "run": [
                        "echo run {application.name} with {user_params:param} on {system.name}"
                    ]
                },
                user_params={
                    "run": [
                        UserParamConfig(
                            name="param=",
                            description="sample parameter",
                            default_value="default",
                            values=["val1", "val2", "val3"],
                            alias="param",
                        )
                    ]
                },
            ),
            SystemConfig(
                name="test_system",
                description="Test system",
                data_transfer=SSHConfig(
                    protocol="ssh",
                    username="username",
                    password="password",
                    hostname="localhost",
                    port="8022",
                ),
                commands={"run": ["sleep 100"]},
            ),
            [
                ExecutionCase(
                    args=[],
                    exit_code=MiddlewareExitCode.SUCCESS,
                    output="""Generating commands to execute
Trying to establish connection with 'localhost:8022' - 90 retries every 15.0 seconds .
Running: echo run test_application with param=default on test_system
Shutting down sequence...
Stopping test_system... (It could take few seconds)
test_system stopped successfully.\n""",
                )
            ],
        ],
    ],
)
def test_application_run(
    application_config: ApplicationConfig,
    system_config: SystemConfig,
    executions: List[ExecutionCase],
    tmpdir: Any,
    cli_runner: CliRunner,
    monkeypatch: Any,
    payload_path_or_none: Path,
) -> None:
    """Test application command execution."""
    for execution in executions:
        init_execution_test(monkeypatch, tmpdir, application_config, system_config)

        if payload_path_or_none:
            with open(payload_path_or_none, "w", encoding="utf-8") as payload_file:
                write_system_payload_config(
                    payload_file, application_config, system_config
                )

            result = cli_runner.invoke(
                run_cmd,
                args=["--config", str(payload_path_or_none)],
            )
        else:
            result = cli_runner.invoke(
                run_cmd,
                args=["-n", application_config["name"], "-s", system_config["name"]]
                + execution["args"],
            )

        assert result.stdout == execution["output"]
        assert result.exit_code == execution["exit_code"]


@pytest.mark.parametrize(
    "cmdline,error_pattern",
    [
        [
            "--config {payload} -s test_system",
            "when --config is set, the following parameters should not be provided",
        ],
        [
            "--config {payload} -n test_application",
            "when --config is set, the following parameters should not be provided",
        ],
        [
            "--config {payload} -p mypar:3",
            "when --config is set, the following parameters should not be provided",
        ],
        [
            "-p mypar:3",
            "when --config is not set, the following parameters are required",
        ],
        ["-s test_system", "when --config is not set, --name is required"],
        ["-n test_application", "when --config is not set, --system is required"],
    ],
)
def test_application_run_invalid_param_combinations(
    cmdline: str,
    error_pattern: str,
    cli_runner: CliRunner,
    monkeypatch: Any,
    tmp_path: Any,
    tmpdir: Any,
) -> None:
    """Test that invalid combinations arguments result in error as expected."""
    application_config = ApplicationConfig(
        name="test_application",
        description="Test application",
        supported_systems=["test_system"],
        build_dir="build",
        commands={"build": ["echo build {application.name} with {user_params:0}"]},
        user_params={
            "build": [
                UserParamConfig(
                    name="param",
                    description="sample parameter",
                    default_value="default",
                    values=["val1", "val2", "val3"],
                )
            ]
        },
    )
    system_config = SystemConfig(
        name="test_system",
        description="Test system",
        data_transfer=LocalProtocolConfig(protocol="local"),
        commands={"run": ["echo run {application.name} on {system.name}"]},
    )

    init_execution_test(monkeypatch, tmpdir, application_config, system_config)

    payload_file = tmp_path / "payload.json"
    payload_file.write_text("dummy")
    result = cli_runner.invoke(
        run_cmd,
        args=cmdline.format(payload=payload_file).split(),
    )
    found = re.search(error_pattern, result.stdout)
    assert found, f"Cannot find pattern: [{error_pattern}] in \n[\n{result.stdout}\n]"


@pytest.mark.parametrize(
    "payload,expected",
    [
        pytest.param(
            {"arguments": {}},
            None,
            marks=pytest.mark.xfail(reason="no system 'id''", strict=True),
        ),
        pytest.param(
            {"id": "testsystem"},
            None,
            marks=pytest.mark.xfail(reason="no arguments object", strict=True),
        ),
        (
            {"id": "testsystem", "arguments": {"application": "testapp"}},
            ("testsystem", "testapp", [], [], [], None),
        ),
        (
            {
                "id": "testsystem",
                "arguments": {"application": "testapp", "par1": "val1"},
            },
            ("testsystem", "testapp", ["par1=val1"], [], [], None),
        ),
        (
            {
                "id": "testsystem",
                "arguments": {"application": "testapp", "application/par1": "val1"},
            },
            ("testsystem", "testapp", ["par1=val1"], [], [], None),
        ),
        (
            {
                "id": "testsystem",
                "arguments": {"application": "testapp", "system/par1": "val1"},
            },
            ("testsystem", "testapp", [], ["par1=val1"], [], None),
        ),
        (
            {
                "id": "testsystem",
                "arguments": {"application": "testapp", "deploy/par1": "val1"},
            },
            ("testsystem", "testapp", [], [], ["par1"], None),
        ),
        (
            {
                "id": "testsystem",
                "arguments": {
                    "application": "testapp",
                    "appar1": "val1",
                    "application/appar2": "val2",
                    "system/syspar1": "val3",
                    "deploy/depploypar1": "val4",
                    "application/appar3": "val5",
                    "system/syspar2": "val6",
                    "deploy/depploypar2": "val7",
                },
            },
            (
                "testsystem",
                "testapp",
                ["appar1=val1", "appar2=val2", "appar3=val5"],
                ["syspar1=val3", "syspar2=val6"],
                ["depploypar1", "depploypar2"],
                None,
            ),
        ),
    ],
)
def test_parse_payload_run_config(payload: dict, expected: tuple) -> None:
    """Test parsing of the JSON payload for the run_config command."""
    assert parse_payload_run_config(payload) == expected


def test_application_run_report(
    tmpdir: Any,
    cli_runner: CliRunner,
    monkeypatch: Any,
) -> None:
    """Test flag '--report' of command 'application run'."""
    app_metrics = {"app_metric": 3.14}
    app_metrics_b64 = base64.b64encode(json.dumps(app_metrics).encode("utf-8"))
    application_config = ApplicationConfig(
        name="test_application",
        description="Test application",
        supported_systems=["test_system"],
        build_dir="build",
        commands={"build": ["echo build {application.name} with {user_params:0}"]},
        user_params={
            "build": [
                UserParamConfig(
                    name="param",
                    description="sample parameter",
                    default_value="default",
                    values=["val1", "val2", "val3"],
                ),
                UserParamConfig(
                    name="p2",
                    description="another parameter, not overridden",
                    default_value="the-right-choice",
                    values=["the-right-choice", "the-bad-choice"],
                ),
            ]
        },
    )
    system_config = SystemConfig(
        name="test_system",
        description="Test system",
        data_transfer=LocalProtocolConfig(protocol="local"),
        commands={
            "run": [
                "echo run {application.name} on {system.name}",
                f"echo build <{Base64OutputParser.TAG_NAME}>{app_metrics_b64.decode('utf-8')}</{Base64OutputParser.TAG_NAME}>",
            ]
        },
        reporting={
            "regex": {
                "app_name": {
                    "pattern": r"run (.\S*) ",
                    "type": "str",
                },
                "sys_name": {
                    "pattern": r"on (.\S*)",
                    "type": "str",
                },
            }
        },
    )
    report_file = Path(tmpdir) / "test_report.json"
    param_val = "param=val1"
    exit_code = MiddlewareExitCode.SUCCESS

    init_execution_test(monkeypatch, tmpdir, application_config, system_config)

    result = cli_runner.invoke(
        run_cmd,
        args=[
            "-n",
            application_config["name"],
            "-s",
            system_config["name"],
            "--report",
            str(report_file),
            "--param",
            param_val,
        ],
    )
    assert result.exit_code == exit_code
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as file:
        report = json.load(file)

        assert report == {
            "application": {
                "metrics": {"0": {"app_metric": 3.14}},
                "name": "test_application",
                "params": {"param": "val1", "p2": "the-right-choice"},
            },
            "system": {
                "metrics": {"app_name": "test_application", "sys_name": "test_system"},
                "name": "test_system",
                "params": {},
            },
        }


def init_execution_test(
    monkeypatch: Any,
    tmpdir: Any,
    application_config: ApplicationConfig,
    system_config: SystemConfig,
    can_establish_connection: bool = True,
    establish_conection_delay: float = 0,
    remote_app_exit_code: int = 0,
) -> None:
    """Init execution test."""
    application_name = application_config["name"]
    system_name = system_config["name"]

    execute_cmd.params[0].type = click.Choice([application_name])
    execute_cmd.params[1].type = click.Choice([system_name])
    execute_cmd.params[2].type = click.Choice(["build", "run", "some_command"])

    run_cmd.params[0].type = click.Choice([application_name])
    run_cmd.params[1].type = click.Choice([system_name])

    if "config_location" not in application_config:
        application_path = Path(tmpdir) / "application"
        application_path.mkdir()
        application_config["config_location"] = application_path

        # this file could be used as deploy parameter value or
        # as deploy parameter in application configuration
        sample_file = application_path / "sample_file"
        sample_file.touch()
    monkeypatch.setattr(
        "aiet.backend.application.get_available_applications",
        MagicMock(return_value=[Application(application_config)]),
    )

    ssh_protocol_mock = MagicMock(spec=SSHProtocol)

    def mock_establish_connection() -> bool:
        """Mock establish connection function."""
        # give some time for the system to start
        time.sleep(establish_conection_delay)
        return can_establish_connection

    ssh_protocol_mock.establish_connection.side_effect = mock_establish_connection
    ssh_protocol_mock.connection_details.return_value = ("localhost", 8022)
    ssh_protocol_mock.run.return_value = (
        remote_app_exit_code,
        bytearray(),
        bytearray(),
    )
    monkeypatch.setattr(
        "aiet.backend.protocol.SSHProtocol", MagicMock(return_value=ssh_protocol_mock)
    )

    if "config_location" not in system_config:
        system_path = Path(tmpdir) / "system"
        system_path.mkdir()
        system_config["config_location"] = system_path
    monkeypatch.setattr(
        "aiet.backend.system.get_available_systems",
        MagicMock(return_value=[load_system(system_config)]),
    )

    monkeypatch.setattr("aiet.backend.execution.wait", MagicMock())
