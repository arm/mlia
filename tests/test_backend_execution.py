# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test backend execution module."""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.application import Application
from mlia.backend.common import UserParamConfig
from mlia.backend.config import ApplicationConfig
from mlia.backend.config import SystemConfig
from mlia.backend.execution import ExecutionContext
from mlia.backend.execution import get_application_and_system
from mlia.backend.execution import get_application_by_name_and_system
from mlia.backend.execution import ParamResolver
from mlia.backend.execution import run_application
from mlia.backend.system import load_system


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
    )

    param_resolver = ParamResolver(ctx)
    expected_values = {
        "application.name": "test_application",
        "application.description": "Test application",
        "application.config_dir": str(application_config_location),
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


def test_get_application_by_name_and_system(monkeypatch: Any) -> None:
    """Test exceptional case for get_application_by_name_and_system."""
    monkeypatch.setattr(
        "mlia.backend.execution.get_application",
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
        "mlia.backend.execution.get_system", MagicMock(return_value=None)
    )

    with pytest.raises(ValueError, match="System test_system is not found"):
        get_application_and_system("test_application", "test_system")


def test_run_application() -> None:
    """Test function run_application."""
    ctx = run_application("application_4", [], "System 4", [])

    assert isinstance(ctx, ExecutionContext)
    assert ctx.stderr is not None and not ctx.stderr.decode()
    assert ctx.stdout is not None and ctx.stdout.decode().strip() == "application_4"
