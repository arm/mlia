# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for system backend."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from mlia.backend.common import Command
from mlia.backend.common import ConfigurationException
from mlia.backend.common import Param
from mlia.backend.common import UserParamConfig
from mlia.backend.config import LocalProtocolConfig
from mlia.backend.config import ProtocolConfig
from mlia.backend.config import SSHConfig
from mlia.backend.config import SystemConfig
from mlia.backend.controller import SystemController
from mlia.backend.controller import SystemControllerSingleInstance
from mlia.backend.protocol import LocalProtocol
from mlia.backend.protocol import SSHProtocol
from mlia.backend.protocol import SupportsClose
from mlia.backend.protocol import SupportsDeploy
from mlia.backend.system import ControlledSystem
from mlia.backend.system import get_available_systems
from mlia.backend.system import get_controller
from mlia.backend.system import get_system
from mlia.backend.system import install_system
from mlia.backend.system import load_system
from mlia.backend.system import remove_system
from mlia.backend.system import StandaloneSystem
from mlia.backend.system import System


def dummy_resolver(
    values: Optional[Dict[str, str]] = None
) -> Callable[[str, str, List[Tuple[Optional[str], Param]]], str]:
    """Return dummy parameter resolver implementation."""
    # pylint: disable=unused-argument
    def resolver(
        param: str, cmd: str, param_values: List[Tuple[Optional[str], Param]]
    ) -> str:
        """Implement dummy parameter resolver."""
        return values.get(param, "") if values else ""

    return resolver


def test_get_available_systems() -> None:
    """Test get_available_systems mocking get_resources."""
    available_systems = get_available_systems()
    assert all(isinstance(s, System) for s in available_systems)
    assert len(available_systems) == 4
    assert [str(s) for s in available_systems] == [
        "System 1",
        "System 2",
        "System 4",
        "System 6",
    ]


def test_get_system() -> None:
    """Test get_system."""
    system1 = get_system("System 1")
    assert isinstance(system1, ControlledSystem)
    assert system1.connectable is True
    assert system1.connection_details() == ("localhost", 8021)
    assert system1.name == "System 1"

    system2 = get_system("System 2")
    # check that comparison with object of another type returns false
    assert system1 != 42
    assert system1 != system2

    system = get_system("Unknown system")
    assert system is None


@pytest.mark.parametrize(
    "source, call_count, exception_type",
    (
        (
            "archives/systems/system1.tar.gz",
            0,
            pytest.raises(Exception, match="Systems .* are already installed"),
        ),
        (
            "archives/systems/system3.tar.gz",
            0,
            pytest.raises(Exception, match="Unable to read system definition"),
        ),
        (
            "backends/systems/system1",
            0,
            pytest.raises(Exception, match="Systems .* are already installed"),
        ),
        (
            "backends/systems/system3",
            0,
            pytest.raises(Exception, match="Unable to read system definition"),
        ),
        ("unknown_path", 0, pytest.raises(Exception, match="Unable to read")),
        (
            "various/systems/system_with_empty_config",
            0,
            pytest.raises(Exception, match="No system definition found"),
        ),
        ("various/systems/system_with_valid_config", 1, does_not_raise()),
    ),
)
def test_install_system(
    monkeypatch: Any,
    test_resources_path: Path,
    source: str,
    call_count: int,
    exception_type: Any,
) -> None:
    """Test system installation from archive."""
    mock_create_destination_and_install = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.system.create_destination_and_install",
        mock_create_destination_and_install,
    )

    with exception_type:
        install_system(test_resources_path / source)

    assert mock_create_destination_and_install.call_count == call_count


def test_remove_system(monkeypatch: Any) -> None:
    """Test system removal."""
    mock_remove_backend = MagicMock()
    monkeypatch.setattr("mlia.backend.system.remove_backend", mock_remove_backend)
    remove_system("some_system_dir")
    mock_remove_backend.assert_called_once()


def test_system(monkeypatch: Any) -> None:
    """Test the System class."""
    config = SystemConfig(name="System 1")
    monkeypatch.setattr("mlia.backend.system.ProtocolFactory", MagicMock())
    system = System(config)
    assert str(system) == "System 1"
    assert system.name == "System 1"


def test_system_with_empty_parameter_name() -> None:
    """Test that configuration fails if parameter name is empty."""
    bad_config = SystemConfig(
        name="System 1",
        commands={"run": ["run"]},
        user_params={"run": [{"name": "", "values": ["1", "2", "3"]}]},
    )
    with pytest.raises(Exception, match="Parameter has an empty 'name' attribute."):
        System(bad_config)


def test_system_standalone_run() -> None:
    """Test run operation for standalone system."""
    system = get_system("System 4")
    assert isinstance(system, StandaloneSystem)

    with pytest.raises(
        ConfigurationException, match="System .* does not support connections"
    ):
        system.connection_details()

    with pytest.raises(
        ConfigurationException, match="System .* does not support connections"
    ):
        system.establish_connection()

    assert system.connectable is False

    system.run("echo 'application run'")


@pytest.mark.parametrize(
    "system_name, expected_value", [("System 1", True), ("System 4", False)]
)
def test_system_supports_deploy(system_name: str, expected_value: bool) -> None:
    """Test system property supports_deploy."""
    system = get_system(system_name)
    if system is None:
        pytest.fail("Unable to get system {}".format(system_name))
    assert system.supports_deploy == expected_value


@pytest.mark.parametrize(
    "mock_protocol",
    [
        MagicMock(spec=SSHProtocol),
        MagicMock(
            spec=SSHProtocol,
            **{"close.side_effect": ValueError("Unable to close protocol")}
        ),
        MagicMock(spec=LocalProtocol),
    ],
)
def test_system_start_and_stop(monkeypatch: Any, mock_protocol: MagicMock) -> None:
    """Test system start, run commands and stop."""
    monkeypatch.setattr(
        "mlia.backend.system.ProtocolFactory.get_protocol",
        MagicMock(return_value=mock_protocol),
    )

    system = get_system("System 1")
    if system is None:
        pytest.fail("Unable to get system")
    assert isinstance(system, ControlledSystem)

    with pytest.raises(Exception, match="System has not been started"):
        system.stop()

    assert not system.is_running()
    assert system.get_output() == ("", "")
    system.start(["sleep 10"], False)
    assert system.is_running()
    system.stop(wait=True)
    assert not system.is_running()
    assert system.get_output() == ("", "")

    if isinstance(mock_protocol, SupportsClose):
        mock_protocol.close.assert_called_once()

    if isinstance(mock_protocol, SSHProtocol):
        system.establish_connection()


def test_system_start_no_config_location() -> None:
    """Test that system without config location could not start."""
    system = load_system(
        SystemConfig(
            name="test",
            data_transfer=SSHConfig(
                protocol="ssh",
                username="user",
                password="user",
                hostname="localhost",
                port="123",
            ),
        )
    )

    assert isinstance(system, ControlledSystem)
    with pytest.raises(
        ConfigurationException, match="System test has wrong config location"
    ):
        system.start(["sleep 100"])


@pytest.mark.parametrize(
    "config, expected_class, expected_error",
    [
        (
            SystemConfig(
                name="test",
                data_transfer=SSHConfig(
                    protocol="ssh",
                    username="user",
                    password="user",
                    hostname="localhost",
                    port="123",
                ),
            ),
            ControlledSystem,
            does_not_raise(),
        ),
        (
            SystemConfig(
                name="test", data_transfer=LocalProtocolConfig(protocol="local")
            ),
            StandaloneSystem,
            does_not_raise(),
        ),
        (
            SystemConfig(
                name="test",
                data_transfer=ProtocolConfig(protocol="cool_protocol"),  # type: ignore
            ),
            None,
            pytest.raises(
                Exception, match="Unsupported execution type for protocol cool_protocol"
            ),
        ),
    ],
)
def test_load_system(
    config: SystemConfig, expected_class: type, expected_error: Any
) -> None:
    """Test load_system function."""
    if not expected_class:
        with expected_error:
            load_system(config)
    else:
        system = load_system(config)
        assert isinstance(system, expected_class)


def test_load_system_populate_shared_params() -> None:
    """Test shared parameters population."""
    with pytest.raises(Exception, match="All shared parameters should have aliases"):
        load_system(
            SystemConfig(
                name="test_system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                user_params={
                    "shared": [
                        UserParamConfig(
                            name="--shared_param1",
                            description="Shared parameter",
                            values=["1", "2", "3"],
                            default_value="1",
                        )
                    ]
                },
            )
        )

    with pytest.raises(
        Exception, match="All parameters for command run should have aliases"
    ):
        load_system(
            SystemConfig(
                name="test_system",
                data_transfer=LocalProtocolConfig(protocol="local"),
                user_params={
                    "shared": [
                        UserParamConfig(
                            name="--shared_param1",
                            description="Shared parameter",
                            values=["1", "2", "3"],
                            default_value="1",
                            alias="shared_param1",
                        )
                    ],
                    "run": [
                        UserParamConfig(
                            name="--run_param1",
                            description="Run specific parameter",
                            values=["1", "2", "3"],
                            default_value="2",
                        )
                    ],
                },
            )
        )
    system0 = load_system(
        SystemConfig(
            name="test_system",
            data_transfer=LocalProtocolConfig(protocol="local"),
            commands={"run": ["run_command"]},
            user_params={
                "shared": [],
                "run": [
                    UserParamConfig(
                        name="--run_param1",
                        description="Run specific parameter",
                        values=["1", "2", "3"],
                        default_value="2",
                        alias="run_param1",
                    )
                ],
            },
        )
    )
    assert len(system0.commands) == 1
    run_command1 = system0.commands["run"]
    assert run_command1 == Command(
        ["run_command"],
        [
            Param(
                "--run_param1",
                "Run specific parameter",
                ["1", "2", "3"],
                "2",
                "run_param1",
            )
        ],
    )

    system1 = load_system(
        SystemConfig(
            name="test_system",
            data_transfer=LocalProtocolConfig(protocol="local"),
            user_params={
                "shared": [
                    UserParamConfig(
                        name="--shared_param1",
                        description="Shared parameter",
                        values=["1", "2", "3"],
                        default_value="1",
                        alias="shared_param1",
                    )
                ],
                "run": [
                    UserParamConfig(
                        name="--run_param1",
                        description="Run specific parameter",
                        values=["1", "2", "3"],
                        default_value="2",
                        alias="run_param1",
                    )
                ],
            },
        )
    )
    assert len(system1.commands) == 2
    build_command1 = system1.commands["build"]
    assert build_command1 == Command(
        [],
        [
            Param(
                "--shared_param1",
                "Shared parameter",
                ["1", "2", "3"],
                "1",
                "shared_param1",
            )
        ],
    )

    run_command1 = system1.commands["run"]
    assert run_command1 == Command(
        [],
        [
            Param(
                "--shared_param1",
                "Shared parameter",
                ["1", "2", "3"],
                "1",
                "shared_param1",
            ),
            Param(
                "--run_param1",
                "Run specific parameter",
                ["1", "2", "3"],
                "2",
                "run_param1",
            ),
        ],
    )

    system2 = load_system(
        SystemConfig(
            name="test_system",
            data_transfer=LocalProtocolConfig(protocol="local"),
            commands={"build": ["build_command"]},
            user_params={
                "shared": [
                    UserParamConfig(
                        name="--shared_param1",
                        description="Shared parameter",
                        values=["1", "2", "3"],
                        default_value="1",
                        alias="shared_param1",
                    )
                ],
                "run": [
                    UserParamConfig(
                        name="--run_param1",
                        description="Run specific parameter",
                        values=["1", "2", "3"],
                        default_value="2",
                        alias="run_param1",
                    )
                ],
            },
        )
    )
    assert len(system2.commands) == 2
    build_command2 = system2.commands["build"]
    assert build_command2 == Command(
        ["build_command"],
        [
            Param(
                "--shared_param1",
                "Shared parameter",
                ["1", "2", "3"],
                "1",
                "shared_param1",
            )
        ],
    )

    run_command2 = system1.commands["run"]
    assert run_command2 == Command(
        [],
        [
            Param(
                "--shared_param1",
                "Shared parameter",
                ["1", "2", "3"],
                "1",
                "shared_param1",
            ),
            Param(
                "--run_param1",
                "Run specific parameter",
                ["1", "2", "3"],
                "2",
                "run_param1",
            ),
        ],
    )


@pytest.mark.parametrize(
    "mock_protocol, expected_call_count",
    [(MagicMock(spec=SupportsDeploy), 1), (MagicMock(), 0)],
)
def test_system_deploy_data(
    monkeypatch: Any, mock_protocol: MagicMock, expected_call_count: int
) -> None:
    """Test deploy data functionality."""
    monkeypatch.setattr(
        "mlia.backend.system.ProtocolFactory.get_protocol",
        MagicMock(return_value=mock_protocol),
    )

    system = ControlledSystem(SystemConfig(name="test"))
    system.deploy(Path("some_file"), "some_dest")

    assert mock_protocol.deploy.call_count == expected_call_count


@pytest.mark.parametrize(
    "single_instance, controller_class",
    ((False, SystemController), (True, SystemControllerSingleInstance)),
)
def test_get_controller(single_instance: bool, controller_class: type) -> None:
    """Test function get_controller."""
    controller = get_controller(single_instance)
    assert isinstance(controller, controller_class)
