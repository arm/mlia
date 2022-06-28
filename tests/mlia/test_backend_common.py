# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use,protected-access
"""Tests for the common backend module."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import IO
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from unittest.mock import MagicMock

import pytest

from mlia.backend.application import Application
from mlia.backend.common import Backend
from mlia.backend.common import BaseBackendConfig
from mlia.backend.common import Command
from mlia.backend.common import ConfigurationException
from mlia.backend.common import load_config
from mlia.backend.common import Param
from mlia.backend.common import parse_raw_parameter
from mlia.backend.common import remove_backend
from mlia.backend.config import ApplicationConfig
from mlia.backend.config import UserParamConfig
from mlia.backend.execution import ExecutionContext
from mlia.backend.execution import ParamResolver
from mlia.backend.system import System


@pytest.mark.parametrize(
    "directory_name, expected_exception",
    (
        ("some_dir", does_not_raise()),
        (None, pytest.raises(Exception, match="No directory name provided")),
    ),
)
def test_remove_backend(
    monkeypatch: Any, directory_name: str, expected_exception: Any
) -> None:
    """Test remove_backend function."""
    mock_remove_resource = MagicMock()
    monkeypatch.setattr("mlia.backend.common.remove_resource", mock_remove_resource)

    with expected_exception:
        remove_backend(directory_name, "applications")


@pytest.mark.parametrize(
    "filename, expected_exception",
    (
        ("application_config.json", does_not_raise()),
        (None, pytest.raises(Exception, match="Unable to read config")),
    ),
)
def test_load_config(
    filename: str, expected_exception: Any, test_resources_path: Path, monkeypatch: Any
) -> None:
    """Test load_config."""
    with expected_exception:
        configs: List[Optional[Union[Path, IO[bytes]]]] = (
            [None]
            if not filename
            else [
                # Ignore pylint warning as 'with' can't be used inside of a
                # generator expression.
                # pylint: disable=consider-using-with
                open(test_resources_path / filename, "rb"),
                test_resources_path / filename,
            ]
        )
        for config in configs:
            json_mock = MagicMock()
            monkeypatch.setattr("mlia.backend.common.json.load", json_mock)
            load_config(config)
            json_mock.assert_called_once()


class TestBackend:
    """Test Backend class."""

    def test___repr__(self) -> None:
        """Test the representation of Backend instance."""
        backend = Backend(
            BaseBackendConfig(name="Testing name", description="Testing description")
        )
        assert str(backend) == "Testing name"

    def test__eq__(self) -> None:
        """Test equality method with different cases."""
        backend1 = Backend(BaseBackendConfig(name="name", description="description"))
        backend1.commands = {"command": Command(["command"])}

        backend2 = Backend(BaseBackendConfig(name="name", description="description"))
        backend2.commands = {"command": Command(["command"])}

        backend3 = Backend(
            BaseBackendConfig(
                name="Ben", description="This is not the Backend you are looking for"
            )
        )
        backend3.commands = {"wave": Command(["wave hand"])}

        backend4 = "Foo"  # checking not isinstance(backend4, Backend)

        assert backend1 == backend2
        assert backend1 != backend3
        assert backend1 != backend4

    @pytest.mark.parametrize(
        "parameter, valid",
        [
            ("--choice-param dummy_value_1", True),
            ("--choice-param wrong_value", False),
            ("--open-param something", True),
            ("--wrong-param value", False),
        ],
    )
    def test_validate_parameter(
        self, parameter: str, valid: bool, test_resources_path: Path
    ) -> None:
        """Test validate_parameter."""
        config = cast(
            List[ApplicationConfig],
            load_config(test_resources_path / "hello_world.json"),
        )
        # The application configuration is a list of configurations so we need
        # only the first one
        # Exercise the validate_parameter test using the Application classe which
        # inherits from Backend.
        application = Application(config[0])
        assert application.validate_parameter("run", parameter) == valid

    def test_validate_parameter_with_invalid_command(
        self, test_resources_path: Path
    ) -> None:
        """Test validate_parameter with an invalid command_name."""
        config = cast(
            List[ApplicationConfig],
            load_config(test_resources_path / "hello_world.json"),
        )
        application = Application(config[0])
        with pytest.raises(AttributeError) as err:
            # command foo does not exist, so raise an error
            application.validate_parameter("foo", "bar")
        assert "Unknown command: 'foo'" in str(err.value)

    def test_build_command(self, monkeypatch: Any) -> None:
        """Test command building."""
        config = {
            "name": "test",
            "commands": {
                "build": ["build {user_params:0} {user_params:1}"],
                "run": ["run {user_params:0}"],
                "post_run": ["post_run {application_params:0} on {system_params:0}"],
                "some_command": ["Command with {variables:var_A}"],
                "empty_command": [""],
            },
            "user_params": {
                "build": [
                    {
                        "name": "choice_param_0=",
                        "values": [1, 2, 3],
                        "default_value": 1,
                    },
                    {"name": "choice_param_1", "values": [3, 4, 5], "default_value": 3},
                    {"name": "choice_param_3", "values": [6, 7, 8]},
                ],
                "run": [{"name": "flag_param_0"}],
            },
            "variables": {"var_A": "value for variable A"},
        }

        monkeypatch.setattr("mlia.backend.system.ProtocolFactory", MagicMock())
        application, system = Application(config), System(config)  # type: ignore
        context = ExecutionContext(
            app=application,
            app_params=[],
            system=system,
            system_params=[],
            custom_deploy_data=[],
        )

        param_resolver = ParamResolver(context)

        cmd = application.build_command(
            "build", ["choice_param_0=2", "choice_param_1=4"], param_resolver
        )
        assert cmd == ["build choice_param_0=2 choice_param_1 4"]

        cmd = application.build_command("build", ["choice_param_0=2"], param_resolver)
        assert cmd == ["build choice_param_0=2 choice_param_1 3"]

        cmd = application.build_command(
            "build", ["choice_param_0=2", "choice_param_3=7"], param_resolver
        )
        assert cmd == ["build choice_param_0=2 choice_param_1 3"]

        with pytest.raises(
            ConfigurationException, match="Command 'foo' could not be found."
        ):
            application.build_command("foo", [""], param_resolver)

        cmd = application.build_command("some_command", [], param_resolver)
        assert cmd == ["Command with value for variable A"]

        cmd = application.build_command("empty_command", [], param_resolver)
        assert cmd == [""]

    @pytest.mark.parametrize("class_", [Application, System])
    def test_build_command_unknown_variable(self, class_: type) -> None:
        """Test that unable to construct backend with unknown variable."""
        with pytest.raises(Exception, match="Unknown variable var1"):
            config = {"name": "test", "commands": {"run": ["run {variables:var1}"]}}
            class_(config)

    @pytest.mark.parametrize(
        "class_, config, expected_output",
        [
            (
                Application,
                {
                    "name": "test",
                    "commands": {
                        "build": ["build {user_params:0} {user_params:1}"],
                        "run": ["run {user_params:0}"],
                    },
                    "user_params": {
                        "build": [
                            {
                                "name": "choice_param_0=",
                                "values": ["a", "b", "c"],
                                "default_value": "a",
                                "alias": "param_1",
                            },
                            {
                                "name": "choice_param_1",
                                "values": ["a", "b", "c"],
                                "default_value": "a",
                                "alias": "param_2",
                            },
                            {"name": "choice_param_3", "values": ["a", "b", "c"]},
                        ],
                        "run": [{"name": "flag_param_0"}],
                    },
                },
                [
                    (
                        "b",
                        Param(
                            name="choice_param_0=",
                            description="",
                            values=["a", "b", "c"],
                            default_value="a",
                            alias="param_1",
                        ),
                    ),
                    (
                        "a",
                        Param(
                            name="choice_param_1",
                            description="",
                            values=["a", "b", "c"],
                            default_value="a",
                            alias="param_2",
                        ),
                    ),
                    (
                        "c",
                        Param(
                            name="choice_param_3",
                            description="",
                            values=["a", "b", "c"],
                        ),
                    ),
                ],
            ),
            (System, {"name": "test"}, []),
        ],
    )
    def test_resolved_parameters(
        self,
        monkeypatch: Any,
        class_: type,
        config: Dict,
        expected_output: List[Tuple[Optional[str], Param]],
    ) -> None:
        """Test command building."""
        monkeypatch.setattr("mlia.backend.system.ProtocolFactory", MagicMock())
        backend = class_(config)

        params = backend.resolved_parameters(
            "build", ["choice_param_0=b", "choice_param_3=c"]
        )
        assert params == expected_output

    @pytest.mark.parametrize(
        ["param_name", "user_param", "expected_value"],
        [
            (
                "test_name",
                "test_name=1234",
                "1234",
            ),  # optional parameter using '='
            (
                "test_name",
                "test_name 1234",
                "1234",
            ),  # optional parameter using ' '
            ("test_name", "test_name", None),  # flag
            (None, "test_name=1234", "1234"),  # positional parameter
        ],
    )
    def test_resolved_user_parameters(
        self, param_name: str, user_param: str, expected_value: str
    ) -> None:
        """Test different variants to provide user parameters."""
        # A dummy config providing one backend config
        config = {
            "name": "test_backend",
            "commands": {
                "test": ["user_param:test_param"],
            },
            "user_params": {
                "test": [UserParamConfig(name=param_name, alias="test_name")],
            },
        }
        backend = Backend(cast(BaseBackendConfig, config))
        params = backend.resolved_parameters(
            command_name="test", user_params=[user_param]
        )
        assert len(params) == 1
        value, param = params[0]
        assert param_name == param.name
        assert expected_value == value

    @pytest.mark.parametrize(
        "input_param,expected",
        [
            ("--param=1", ("--param", "1")),
            ("--param 1", ("--param", "1")),
            ("--flag", ("--flag", None)),
        ],
    )
    def test__parse_raw_parameter(
        self, input_param: str, expected: Tuple[str, Optional[str]]
    ) -> None:
        """Test internal method of parsing a single raw parameter."""
        assert parse_raw_parameter(input_param) == expected


class TestParam:
    """Test Param class."""

    def test__eq__(self) -> None:
        """Test equality method with different cases."""
        param1 = Param(name="test", description="desc", values=["values"])
        param2 = Param(name="test", description="desc", values=["values"])
        param3 = Param(name="test1", description="desc", values=["values"])
        param4 = object()

        assert param1 == param2
        assert param1 != param3
        assert param1 != param4

    def test_get_details(self) -> None:
        """Test get_details() method."""
        param1 = Param(name="test", description="desc", values=["values"])
        assert param1.get_details() == {
            "name": "test",
            "values": ["values"],
            "description": "desc",
        }

    def test_invalid(self) -> None:
        """Test invalid use cases for the Param class."""
        with pytest.raises(
            ConfigurationException,
            match="Either name, alias or both must be set to identify a parameter.",
        ):
            Param(name=None, description="desc", values=["values"])


class TestCommand:
    """Test Command class."""

    def test_get_details(self) -> None:
        """Test get_details() method."""
        param1 = Param(name="test", description="desc", values=["values"])
        command1 = Command(command_strings=["echo test"], params=[param1])
        assert command1.get_details() == {
            "command_strings": ["echo test"],
            "user_params": [
                {"name": "test", "values": ["values"], "description": "desc"}
            ],
        }

    def test__eq__(self) -> None:
        """Test equality method with different cases."""
        param1 = Param("test", "desc", ["values"])
        param2 = Param("test1", "desc1", ["values1"])
        command1 = Command(command_strings=["echo test"], params=[param1])
        command2 = Command(command_strings=["echo test"], params=[param1])
        command3 = Command(command_strings=["echo test"])
        command4 = Command(command_strings=["echo test"], params=[param2])
        command5 = object()

        assert command1 == command2
        assert command1 != command3
        assert command1 != command4
        assert command1 != command5

    @pytest.mark.parametrize(
        "params, expected_error",
        [
            [[], does_not_raise()],
            [[Param("param", "param description", [])], does_not_raise()],
            [
                [
                    Param("param", "param description", [], None, "alias"),
                    Param("param", "param description", [], None),
                ],
                does_not_raise(),
            ],
            [
                [
                    Param("param1", "param1 description", [], None, "alias1"),
                    Param("param2", "param2 description", [], None, "alias2"),
                ],
                does_not_raise(),
            ],
            [
                [
                    Param("param", "param description", [], None, "alias"),
                    Param("param", "param description", [], None, "alias"),
                ],
                pytest.raises(ConfigurationException, match="Non unique aliases alias"),
            ],
            [
                [
                    Param("alias", "param description", [], None, "alias1"),
                    Param("param", "param description", [], None, "alias"),
                ],
                pytest.raises(
                    ConfigurationException,
                    match="Aliases .* could not be used as parameter name",
                ),
            ],
            [
                [
                    Param("alias", "param description", [], None, "alias"),
                    Param("param1", "param1 description", [], None, "alias1"),
                ],
                does_not_raise(),
            ],
            [
                [
                    Param("alias", "param description", [], None, "alias"),
                    Param("alias", "param1 description", [], None, "alias1"),
                ],
                pytest.raises(
                    ConfigurationException,
                    match="Aliases .* could not be used as parameter name",
                ),
            ],
            [
                [
                    Param("param1", "param1 description", [], None, "alias1"),
                    Param("param2", "param2 description", [], None, "alias1"),
                    Param("param3", "param3 description", [], None, "alias2"),
                    Param("param4", "param4 description", [], None, "alias2"),
                ],
                pytest.raises(
                    ConfigurationException, match="Non unique aliases alias1, alias2"
                ),
            ],
        ],
    )
    def test_validate_params(self, params: List[Param], expected_error: Any) -> None:
        """Test command validation function."""
        with expected_error:
            Command([], params)
