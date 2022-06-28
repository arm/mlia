# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use
"""Tests for the application backend."""
from collections import Counter
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import List
from unittest.mock import MagicMock

import pytest

from mlia.backend.application import Application
from mlia.backend.application import get_application
from mlia.backend.application import get_available_application_directory_names
from mlia.backend.application import get_available_applications
from mlia.backend.application import get_unique_application_names
from mlia.backend.application import install_application
from mlia.backend.application import load_applications
from mlia.backend.application import remove_application
from mlia.backend.common import Command
from mlia.backend.common import DataPaths
from mlia.backend.common import Param
from mlia.backend.common import UserParamConfig
from mlia.backend.config import ApplicationConfig
from mlia.backend.config import ExtendedApplicationConfig
from mlia.backend.config import NamedExecutionConfig


def test_get_available_application_directory_names() -> None:
    """Test get_available_applicationss mocking get_resources."""
    directory_names = get_available_application_directory_names()
    assert Counter(directory_names) == Counter(
        [
            "application1",
            "application2",
            "application4",
            "application5",
            "application6",
        ]
    )


def test_get_available_applications() -> None:
    """Test get_available_applicationss mocking get_resources."""
    available_applications = get_available_applications()

    assert all(isinstance(s, Application) for s in available_applications)
    assert all(s != 42 for s in available_applications)
    assert len(available_applications) == 10
    # application_5 has multiply items with multiply supported systems
    assert [str(s) for s in available_applications] == [
        "application_1",
        "application_2",
        "application_4",
        "application_5",
        "application_5",
        "application_5A",
        "application_5A",
        "application_5B",
        "application_5B",
        "application_6",
    ]


def test_get_unique_application_names() -> None:
    """Test get_unique_application_names."""
    unique_names = get_unique_application_names()

    assert all(isinstance(s, str) for s in unique_names)
    assert all(s for s in unique_names)
    assert sorted(unique_names) == [
        "application_1",
        "application_2",
        "application_4",
        "application_5",
        "application_5A",
        "application_5B",
        "application_6",
    ]


def test_get_application() -> None:
    """Test get_application mocking get_resoures."""
    application = get_application("application_1")
    if len(application) != 1:
        pytest.fail("Unable to get application")
    assert application[0].name == "application_1"

    application = get_application("unknown application")
    assert len(application) == 0


@pytest.mark.parametrize(
    "source, call_count, expected_exception",
    (
        (
            "archives/applications/application1.tar.gz",
            0,
            pytest.raises(
                Exception, match=r"Applications \[application_1\] are already installed"
            ),
        ),
        (
            "various/applications/application_with_empty_config",
            0,
            pytest.raises(Exception, match="No application definition found"),
        ),
        (
            "various/applications/application_with_wrong_config1",
            0,
            pytest.raises(Exception, match="Unable to read application definition"),
        ),
        (
            "various/applications/application_with_wrong_config2",
            0,
            pytest.raises(Exception, match="Unable to read application definition"),
        ),
        (
            "various/applications/application_with_wrong_config3",
            0,
            pytest.raises(Exception, match="Unable to read application definition"),
        ),
        ("various/applications/application_with_valid_config", 1, does_not_raise()),
        (
            "archives/applications/application3.tar.gz",
            0,
            pytest.raises(Exception, match="Unable to read application definition"),
        ),
        (
            "backends/applications/application1",
            0,
            pytest.raises(
                Exception, match=r"Applications \[application_1\] are already installed"
            ),
        ),
        (
            "backends/applications/application3",
            0,
            pytest.raises(Exception, match="Unable to read application definition"),
        ),
    ),
)
def test_install_application(
    monkeypatch: Any,
    test_resources_path: Path,
    source: str,
    call_count: int,
    expected_exception: Any,
) -> None:
    """Test application install from archive."""
    mock_create_destination_and_install = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.application.create_destination_and_install",
        mock_create_destination_and_install,
    )

    with expected_exception:
        install_application(test_resources_path / source)
    assert mock_create_destination_and_install.call_count == call_count


def test_remove_application(monkeypatch: Any) -> None:
    """Test application removal."""
    mock_remove_backend = MagicMock()
    monkeypatch.setattr("mlia.backend.application.remove_backend", mock_remove_backend)

    remove_application("some_application_directory")
    mock_remove_backend.assert_called_once()


def test_application_config_without_commands() -> None:
    """Test application config without commands."""
    config = ApplicationConfig(name="application")
    application = Application(config)
    # pylint: disable=use-implicit-booleaness-not-comparison
    assert application.commands == {}


class TestApplication:
    """Test for application class methods."""

    def test___eq__(self) -> None:
        """Test overloaded __eq__ method."""
        config = ApplicationConfig(
            # Application
            supported_systems=["system1", "system2"],
            build_dir="build_dir",
            # inherited from Backend
            name="name",
            description="description",
            commands={},
        )
        application1 = Application(config)
        application2 = Application(config)  # Identical
        assert application1 == application2

        application3 = Application(config)  # changed
        # Change one single attribute so not equal, but same Type
        setattr(application3, "supported_systems", ["somewhere/else"])
        assert application1 != application3

        # different Type
        application4 = "Not the Application you are looking for"
        assert application1 != application4

        application5 = Application(config)
        # supported systems could be in any order
        setattr(application5, "supported_systems", ["system2", "system1"])
        assert application1 == application5

    def test_can_run_on(self) -> None:
        """Test Application can run on."""
        config = ApplicationConfig(name="application", supported_systems=["System-A"])

        application = Application(config)
        assert application.can_run_on("System-A")
        assert not application.can_run_on("System-B")

        applications = get_application("application_1", "System 1")
        assert len(applications) == 1
        assert applications[0].can_run_on("System 1")

    def test_get_deploy_data(self, tmp_path: Path) -> None:
        """Test Application can run on."""
        src, dest = "src", "dest"
        config = ApplicationConfig(
            name="application", deploy_data=[(src, dest)], config_location=tmp_path
        )
        src_path = tmp_path / src
        src_path.mkdir()
        application = Application(config)
        assert application.get_deploy_data() == [DataPaths(src_path, dest)]

    def test_get_deploy_data_no_config_location(self) -> None:
        """Test that getting deploy data fails if no config location provided."""
        with pytest.raises(
            Exception, match="Unable to get application .* config location"
        ):
            Application(ApplicationConfig(name="application")).get_deploy_data()

    def test_unable_to_create_application_without_name(self) -> None:
        """Test that it is not possible to create application without name."""
        with pytest.raises(Exception, match="Name is empty"):
            Application(ApplicationConfig())

    def test_application_config_without_commands(self) -> None:
        """Test application config without commands."""
        config = ApplicationConfig(name="application")
        application = Application(config)
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert application.commands == {}

    @pytest.mark.parametrize(
        "config, expected_params",
        (
            (
                ApplicationConfig(
                    name="application",
                    commands={"command": ["cmd {user_params:0} {user_params:1}"]},
                    user_params={
                        "command": [
                            UserParamConfig(
                                name="--param1", description="param1", alias="param1"
                            ),
                            UserParamConfig(
                                name="--param2", description="param2", alias="param2"
                            ),
                        ]
                    },
                ),
                [Param("--param1", "param1"), Param("--param2", "param2")],
            ),
            (
                ApplicationConfig(
                    name="application",
                    commands={"command": ["cmd {user_params:param1} {user_params:1}"]},
                    user_params={
                        "command": [
                            UserParamConfig(
                                name="--param1", description="param1", alias="param1"
                            ),
                            UserParamConfig(
                                name="--param2", description="param2", alias="param2"
                            ),
                        ]
                    },
                ),
                [Param("--param1", "param1"), Param("--param2", "param2")],
            ),
            (
                ApplicationConfig(
                    name="application",
                    commands={"command": ["cmd {user_params:param1}"]},
                    user_params={
                        "command": [
                            UserParamConfig(
                                name="--param1", description="param1", alias="param1"
                            ),
                            UserParamConfig(
                                name="--param2", description="param2", alias="param2"
                            ),
                        ]
                    },
                ),
                [Param("--param1", "param1")],
            ),
        ),
    )
    def test_remove_unused_params(
        self, config: ApplicationConfig, expected_params: List[Param]
    ) -> None:
        """Test mod remove_unused_parameter."""
        application = Application(config)
        application.remove_unused_params()
        assert application.commands["command"].params == expected_params


@pytest.mark.parametrize(
    "config, expected_error",
    (
        (
            ExtendedApplicationConfig(name="application"),
            pytest.raises(Exception, match="No supported systems definition provided"),
        ),
        (
            ExtendedApplicationConfig(
                name="application", supported_systems=[NamedExecutionConfig(name="")]
            ),
            pytest.raises(
                Exception,
                match="Unable to read supported system definition, name is missed",
            ),
        ),
        (
            ExtendedApplicationConfig(
                name="application",
                supported_systems=[
                    NamedExecutionConfig(
                        name="system",
                        commands={"command": ["cmd"]},
                        user_params={"command": [UserParamConfig(name="param")]},
                    )
                ],
                commands={"command": ["cmd {user_params:0}"]},
                user_params={"command": [UserParamConfig(name="param")]},
            ),
            pytest.raises(
                Exception, match="Default parameters for command .* should have aliases"
            ),
        ),
        (
            ExtendedApplicationConfig(
                name="application",
                supported_systems=[
                    NamedExecutionConfig(
                        name="system",
                        commands={"command": ["cmd"]},
                        user_params={"command": [UserParamConfig(name="param")]},
                    )
                ],
                commands={"command": ["cmd {user_params:0}"]},
                user_params={"command": [UserParamConfig(name="param", alias="param")]},
            ),
            pytest.raises(
                Exception, match="system parameters for command .* should have aliases"
            ),
        ),
    ),
)
def test_load_application_exceptional_cases(
    config: ExtendedApplicationConfig, expected_error: Any
) -> None:
    """Test exceptional cases for application load function."""
    with expected_error:
        load_applications(config)


def test_load_application() -> None:
    """Test application load function.

    The main purpose of this test is to test configuration for application
    for different systems. All configuration should be correctly
    overridden if needed.
    """
    application_5 = get_application("application_5")
    assert len(application_5) == 2

    default_commands = {
        "build": Command(["default build command"]),
        "run": Command(["default run command"]),
    }
    default_variables = {"var1": "value1", "var2": "value2"}

    application_5_0 = application_5[0]
    assert application_5_0.build_dir == "default_build_dir"
    assert application_5_0.supported_systems == ["System 1"]
    assert application_5_0.commands == default_commands
    assert application_5_0.variables == default_variables
    assert application_5_0.lock is False

    application_5_1 = application_5[1]
    assert application_5_1.build_dir == application_5_0.build_dir
    assert application_5_1.supported_systems == ["System 2"]
    assert application_5_1.commands == application_5_1.commands
    assert application_5_1.variables == default_variables

    application_5a = get_application("application_5A")
    assert len(application_5a) == 2

    application_5a_0 = application_5a[0]
    assert application_5a_0.supported_systems == ["System 1"]
    assert application_5a_0.build_dir == "build_5A"
    assert application_5a_0.commands == default_commands
    assert application_5a_0.variables == {"var1": "new value1", "var2": "value2"}
    assert application_5a_0.lock is False

    application_5a_1 = application_5a[1]
    assert application_5a_1.supported_systems == ["System 2"]
    assert application_5a_1.build_dir == "build"
    assert application_5a_1.commands == {
        "build": Command(["default build command"]),
        "run": Command(["run command on system 2"]),
    }
    assert application_5a_1.variables == {"var1": "value1", "var2": "new value2"}
    assert application_5a_1.lock is True

    application_5b = get_application("application_5B")
    assert len(application_5b) == 2

    application_5b_0 = application_5b[0]
    assert application_5b_0.build_dir == "build_5B"
    assert application_5b_0.supported_systems == ["System 1"]
    assert application_5b_0.commands == {
        "build": Command(["default build command with value for var1 System1"], []),
        "run": Command(["default run command with value for var2 System1"]),
    }
    assert "non_used_command" not in application_5b_0.commands

    application_5b_1 = application_5b[1]
    assert application_5b_1.build_dir == "build"
    assert application_5b_1.supported_systems == ["System 2"]
    assert application_5b_1.commands == {
        "build": Command(
            [
                "build command on system 2 with value"
                " for var1 System2 {user_params:param1}"
            ],
            [
                Param(
                    "--param",
                    "Sample command param",
                    ["value1", "value2", "value3"],
                    "value1",
                )
            ],
        ),
        "run": Command(["run command on system 2"], []),
    }
