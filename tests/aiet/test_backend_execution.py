# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use
"""Test backend context module."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest
from sh import CommandNotFound

from aiet.backend.application import Application
from aiet.backend.application import get_application
from aiet.backend.common import ConfigurationException
from aiet.backend.common import DataPaths
from aiet.backend.common import UserParamConfig
from aiet.backend.config import ApplicationConfig
from aiet.backend.config import LocalProtocolConfig
from aiet.backend.config import SystemConfig
from aiet.backend.execution import deploy_data
from aiet.backend.execution import execute_commands_locally
from aiet.backend.execution import ExecutionContext
from aiet.backend.execution import get_application_and_system
from aiet.backend.execution import get_application_by_name_and_system
from aiet.backend.execution import get_file_lock_path
from aiet.backend.execution import get_tool_by_system
from aiet.backend.execution import ParamResolver
from aiet.backend.execution import Reporter
from aiet.backend.execution import wait
from aiet.backend.output_parser import OutputParser
from aiet.backend.system import get_system
from aiet.backend.system import load_system
from aiet.backend.tool import get_tool
from aiet.utils.proc import CommandFailedException


def test_context_param_resolver(tmpdir: Any) -> None:
    """Test parameter resolving."""
    system_config_location = Path(tmpdir) / "system"
    system_config_location.mkdir()

    application_config_location = Path(tmpdir) / "application"
    application_config_location.mkdir()

    ctx = ExecutionContext(
        app=Application(
            ApplicationConfig(
                name="test_application",
                description="Test application",
                config_location=application_config_location,
                build_dir="build-{application.name}-{system.name}",
                commands={
                    "run": [
                        "run_command1 {user_params:0}",
                        "run_command2 {user_params:1}",
                    ]
                },
                variables={"var_1": "value for var_1"},
                user_params={
                    "run": [
                        UserParamConfig(
                            name="--param1",
                            description="Param 1",
                            default_value="123",
                            alias="param_1",
                        ),
                        UserParamConfig(
                            name="--param2", description="Param 2", default_value="456"
                        ),
                        UserParamConfig(
                            name="--param3", description="Param 3", alias="param_3"
                        ),
                        UserParamConfig(
                            name="--param4=",
                            description="Param 4",
                            default_value="456",
                            alias="param_4",
                        ),
                        UserParamConfig(
                            description="Param 5",
                            default_value="789",
                            alias="param_5",
                        ),
                    ]
                },
            )
        ),
        app_params=["--param2=789"],
        system=load_system(
            SystemConfig(
                name="test_system",
                description="Test system",
                config_location=system_config_location,
                build_dir="build",
                data_transfer=LocalProtocolConfig(protocol="local"),
                commands={
                    "build": ["build_command1 {user_params:0}"],
                    "run": ["run_command {application.commands.run:1}"],
                },
                variables={"var_1": "value for var_1"},
                user_params={
                    "build": [
                        UserParamConfig(
                            name="--param1", description="Param 1", default_value="aaa"
                        ),
                        UserParamConfig(name="--param2", description="Param 2"),
                    ]
                },
            )
        ),
        system_params=["--param1=bbb"],
        custom_deploy_data=[],
    )

    param_resolver = ParamResolver(ctx)
    expected_values = {
        "application.name": "test_application",
        "application.description": "Test application",
        "application.config_dir": str(application_config_location),
        "application.build_dir": "{}/build-test_application-test_system".format(
            application_config_location
        ),
        "application.commands.run:0": "run_command1 --param1 123",
        "application.commands.run.params:0": "123",
        "application.commands.run.params:param_1": "123",
        "application.commands.run:1": "run_command2 --param2 789",
        "application.commands.run.params:1": "789",
        "application.variables:var_1": "value for var_1",
        "system.name": "test_system",
        "system.description": "Test system",
        "system.config_dir": str(system_config_location),
        "system.commands.build:0": "build_command1 --param1 bbb",
        "system.commands.run:0": "run_command run_command2 --param2 789",
        "system.commands.build.params:0": "bbb",
        "system.variables:var_1": "value for var_1",
    }

    for param, value in expected_values.items():
        assert param_resolver(param) == value

    assert ctx.build_dir() == Path(
        "{}/build-test_application-test_system".format(application_config_location)
    )

    expected_errors = {
        "application.variables:var_2": pytest.raises(
            Exception, match="Unknown variable var_2"
        ),
        "application.commands.clean:0": pytest.raises(
            Exception, match="Command clean not found"
        ),
        "application.commands.run:2": pytest.raises(
            Exception, match="Invalid index 2 for command run"
        ),
        "application.commands.run.params:5": pytest.raises(
            Exception, match="Invalid parameter index 5 for command run"
        ),
        "application.commands.run.params:param_2": pytest.raises(
            Exception,
            match="No value for parameter with index or alias param_2 of command run",
        ),
        "UNKNOWN": pytest.raises(
            Exception, match="Unable to resolve parameter UNKNOWN"
        ),
        "system.commands.build.params:1": pytest.raises(
            Exception,
            match="No value for parameter with index or alias 1 of command build",
        ),
        "system.commands.build:A": pytest.raises(
            Exception, match="Bad command index A"
        ),
        "system.variables:var_2": pytest.raises(
            Exception, match="Unknown variable var_2"
        ),
    }
    for param, error in expected_errors.items():
        with error:
            param_resolver(param)

    resolved_params = ctx.app.resolved_parameters("run", [])
    expected_user_params = {
        "user_params:0": "--param1 123",
        "user_params:param_1": "--param1 123",
        "user_params:2": "--param3",
        "user_params:param_3": "--param3",
        "user_params:3": "--param4=456",
        "user_params:param_4": "--param4=456",
        "user_params:param_5": "789",
    }
    for param, expected_value in expected_user_params.items():
        assert param_resolver(param, "run", resolved_params) == expected_value

    with pytest.raises(
        Exception, match="Invalid index 5 for user params of command run"
    ):
        param_resolver("user_params:5", "run", resolved_params)

    with pytest.raises(
        Exception, match="No user parameter for command 'run' with alias 'param_2'."
    ):
        param_resolver("user_params:param_2", "run", resolved_params)

    with pytest.raises(Exception, match="Unable to resolve user params"):
        param_resolver("user_params:0", "", resolved_params)

    bad_ctx = ExecutionContext(
        app=Application(
            ApplicationConfig(
                name="test_application",
                config_location=application_config_location,
                build_dir="build-{user_params:0}",
            )
        ),
        app_params=["--param2=789"],
        system=load_system(
            SystemConfig(
                name="test_system",
                description="Test system",
                config_location=system_config_location,
                build_dir="build-{system.commands.run.params:123}",
                data_transfer=LocalProtocolConfig(protocol="local"),
            )
        ),
        system_params=["--param1=bbb"],
        custom_deploy_data=[],
    )
    param_resolver = ParamResolver(bad_ctx)
    with pytest.raises(Exception, match="Unable to resolve user params"):
        bad_ctx.build_dir()


# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    "application_name, soft_lock, sys_lock, lock_dir, expected_error, expected_path",
    (
        (
            "test_application",
            True,
            True,
            Path("/tmp"),
            does_not_raise(),
            Path("/tmp/middleware_test_application_test_system.lock"),
        ),
        (
            "$$test_application$!:",
            True,
            True,
            Path("/tmp"),
            does_not_raise(),
            Path("/tmp/middleware_test_application_test_system.lock"),
        ),
        (
            "test_application",
            True,
            True,
            Path("unknown"),
            pytest.raises(
                Exception, match="Invalid directory unknown for lock files provided"
            ),
            None,
        ),
        (
            "test_application",
            False,
            True,
            Path("/tmp"),
            does_not_raise(),
            Path("/tmp/middleware_test_system.lock"),
        ),
        (
            "test_application",
            True,
            False,
            Path("/tmp"),
            does_not_raise(),
            Path("/tmp/middleware_test_application.lock"),
        ),
        (
            "test_application",
            False,
            False,
            Path("/tmp"),
            pytest.raises(Exception, match="No filename for lock provided"),
            None,
        ),
    ),
)
def test_get_file_lock_path(
    application_name: str,
    soft_lock: bool,
    sys_lock: bool,
    lock_dir: Path,
    expected_error: Any,
    expected_path: Path,
) -> None:
    """Test get_file_lock_path function."""
    with expected_error:
        ctx = ExecutionContext(
            app=Application(ApplicationConfig(name=application_name, lock=soft_lock)),
            app_params=[],
            system=load_system(
                SystemConfig(
                    name="test_system",
                    lock=sys_lock,
                    data_transfer=LocalProtocolConfig(protocol="local"),
                )
            ),
            system_params=[],
            custom_deploy_data=[],
        )
        path = get_file_lock_path(ctx, lock_dir)
        assert path == expected_path


def test_get_application_by_name_and_system(monkeypatch: Any) -> None:
    """Test exceptional case for get_application_by_name_and_system."""
    monkeypatch.setattr(
        "aiet.backend.execution.get_application",
        MagicMock(return_value=[MagicMock(), MagicMock()]),
    )

    with pytest.raises(
        ValueError,
        match="Error during getting application test_application for the "
        "system test_system",
    ):
        get_application_by_name_and_system("test_application", "test_system")


def test_get_application_and_system(monkeypatch: Any) -> None:
    """Test exceptional case for get_application_and_system."""
    monkeypatch.setattr(
        "aiet.backend.execution.get_system", MagicMock(return_value=None)
    )

    with pytest.raises(ValueError, match="System test_system is not found"):
        get_application_and_system("test_application", "test_system")


def test_wait_function(monkeypatch: Any) -> None:
    """Test wait function."""
    sleep_mock = MagicMock()
    monkeypatch.setattr("time.sleep", sleep_mock)
    wait(0.1)
    sleep_mock.assert_called_once()


def test_deployment_execution_context() -> None:
    """Test property 'is_deploy_needed' of the ExecutionContext."""
    ctx = ExecutionContext(
        app=get_application("application_1")[0],
        app_params=[],
        system=get_system("System 1"),
        system_params=[],
    )
    assert not ctx.is_deploy_needed
    deploy_data(ctx)  # should be a NOP

    ctx = ExecutionContext(
        app=get_application("application_1")[0],
        app_params=[],
        system=get_system("System 1"),
        system_params=[],
        custom_deploy_data=[DataPaths(Path("README.md"), ".")],
    )
    assert ctx.is_deploy_needed

    ctx = ExecutionContext(
        app=get_application("application_1")[0],
        app_params=[],
        system=None,
        system_params=[],
    )
    assert not ctx.is_deploy_needed
    with pytest.raises(AssertionError):
        deploy_data(ctx)

    ctx = ExecutionContext(
        app=get_tool("tool_1")[0],
        app_params=[],
        system=None,
        system_params=[],
    )
    assert not ctx.is_deploy_needed
    deploy_data(ctx)  # should be a NOP


@pytest.mark.parametrize(
    ["tool_name", "system_name", "exception"],
    [
        ("vela", "Corstone-300: Cortex-M55+Ethos-U65", None),
        ("unknown tool", "Corstone-300: Cortex-M55+Ethos-U65", ConfigurationException),
        ("vela", "unknown system", ConfigurationException),
        ("vela", None, ConfigurationException),
    ],
)
def test_get_tool_by_system(
    tool_name: str, system_name: Optional[str], exception: Optional[Any]
) -> None:
    """Test exceptions thrown by function get_tool_by_system()."""

    def test() -> None:
        """Test call of get_tool_by_system()."""
        tool = get_tool_by_system(tool_name, system_name)
        assert tool is not None

    if exception is None:
        test()
    else:
        with pytest.raises(exception):
            test()


class TestExecuteCommandsLocally:
    """Test execute_commands_locally() function."""

    @pytest.mark.parametrize(
        "first_command, exception, expected_output",
        (
            (
                "echo 'hello'",
                None,
                "Running: echo 'hello'\nhello\nRunning: echo 'goodbye'\ngoodbye\n",
            ),
            (
                "non-existent-command",
                CommandNotFound,
                "Running: non-existent-command\n",
            ),
            ("false", CommandFailedException, "Running: false\n"),
        ),
        ids=(
            "runs_multiple_commands",
            "stops_executing_on_non_existent_command",
            "stops_executing_when_command_exits_with_error_code",
        ),
    )
    def test_execution(
        self,
        first_command: str,
        exception: Any,
        expected_output: str,
        test_resources_path: Path,
        capsys: Any,
    ) -> None:
        """Test expected behaviour of the function."""
        commands = [first_command, "echo 'goodbye'"]
        cwd = test_resources_path
        if exception is None:
            execute_commands_locally(commands, cwd)
        else:
            with pytest.raises(exception):
                execute_commands_locally(commands, cwd)

        captured = capsys.readouterr()
        assert captured.out == expected_output

    def test_stops_executing_on_exception(
        self, monkeypatch: Any, test_resources_path: Path
    ) -> None:
        """Ensure commands following an error-exit-code command don't run."""
        # Mock execute_command() function
        execute_command_mock = mock.MagicMock()
        monkeypatch.setattr("aiet.utils.proc.execute_command", execute_command_mock)

        # Mock Command object and assign as return value to execute_command()
        cmd_mock = mock.MagicMock()
        execute_command_mock.return_value = cmd_mock

        # Mock the terminate_command (speed up test)
        terminate_command_mock = mock.MagicMock()
        monkeypatch.setattr("aiet.utils.proc.terminate_command", terminate_command_mock)

        # Mock a thrown Exception and assign to Command().exit_code
        exit_code_mock = mock.PropertyMock(side_effect=Exception("Exception."))
        type(cmd_mock).exit_code = exit_code_mock

        with pytest.raises(Exception, match="Exception."):
            execute_commands_locally(
                ["command_1", "command_2"], cwd=test_resources_path
            )

        # Assert only "command_1" was executed
        assert execute_command_mock.call_count == 1


def test_reporter(tmpdir: Any) -> None:
    """Test class 'Reporter'."""
    ctx = ExecutionContext(
        app=get_application("application_4")[0],
        app_params=["--app=TestApp"],
        system=get_system("System 4"),
        system_params=[],
    )
    assert ctx.system is not None

    class MockParser(OutputParser):
        """Mock implementation of an output parser."""

        def __init__(self, metrics: Dict[str, Any]) -> None:
            """Set up the MockParser."""
            super().__init__(name="test")
            self.metrics = metrics

        def __call__(self, output: bytearray) -> Dict[str, Any]:
            """Return mock metrics (ignoring the given output)."""
            return self.metrics

    metrics = {"Metric": 123, "AnotherMetric": 456}
    reporter = Reporter(
        parsers=[MockParser(metrics={key: val}) for key, val in metrics.items()],
    )
    reporter.parse(bytearray())
    report = reporter.report(ctx)
    assert report["system"]["name"] == ctx.system.name
    assert report["system"]["params"] == {}
    assert report["application"]["name"] == ctx.app.name
    assert report["application"]["params"] == {"--app": "TestApp"}
    assert report["test"]["metrics"] == metrics
    report_file = Path(tmpdir) / "report.json"
    reporter.save(report, report_file)
    assert report_file.is_file()
