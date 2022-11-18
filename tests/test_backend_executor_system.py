# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for system backend."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.executor.common import Command
from mlia.backend.executor.common import ConfigurationException
from mlia.backend.executor.common import Param
from mlia.backend.executor.common import UserParamConfig
from mlia.backend.executor.config import SystemConfig
from mlia.backend.executor.system import get_available_systems
from mlia.backend.executor.system import get_system
from mlia.backend.executor.system import install_system
from mlia.backend.executor.system import load_system
from mlia.backend.executor.system import remove_system
from mlia.backend.executor.system import System


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
    assert isinstance(system1, System)
    assert system1.name == "System 1"

    system2 = get_system("System 2")
    # check that comparison with object of another type returns false
    assert system1 != 42
    assert system1 != system2

    with pytest.raises(
        ConfigurationException, match="System 'Unknown system' not found."
    ):
        get_system("Unknown system")


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
        "mlia.backend.executor.system.create_destination_and_install",
        mock_create_destination_and_install,
    )

    with exception_type:
        install_system(test_resources_path / source)

    assert mock_create_destination_and_install.call_count == call_count


def test_remove_system(monkeypatch: Any) -> None:
    """Test system removal."""
    mock_remove_backend = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.executor.system.remove_backend", mock_remove_backend
    )
    remove_system("some_system_dir")
    mock_remove_backend.assert_called_once()


def test_system() -> None:
    """Test the System class."""
    config = SystemConfig(name="System 1")
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


def test_system_run() -> None:
    """Test run operation for system."""
    system = get_system("System 4")
    assert isinstance(system, System)

    system.run("echo 'application run'")


def test_system_start_no_config_location() -> None:
    """Test that system without config location could not start."""
    system = load_system(SystemConfig(name="test"))

    assert isinstance(system, System)
    with pytest.raises(
        ConfigurationException, match="System has invalid config location: None"
    ):
        system.run("sleep 100")


@pytest.mark.parametrize(
    "config, expected_class, expected_error",
    [
        (
            SystemConfig(name="test"),
            System,
            does_not_raise(),
        ),
        (SystemConfig(), None, pytest.raises(ConfigurationException)),
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
    assert len(system1.commands) == 1

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
        [],
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
