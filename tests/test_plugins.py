# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for plugin interface."""
import sys
import typing
from dataclasses import dataclass
from os import PathLike
from typing import Union
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.plugins.plugins import call_entry_points
from mlia.plugins.plugins import logger

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata


@dataclass
class MockEntrypoints:
    """Collection of stub entrypoints to load."""

    metadata_entry_points: MagicMock

    first_plugin: MagicMock
    second_plugin: MagicMock


@dataclass
class MockLogging:
    """Collection of stub logging methods."""

    mock_debug: MagicMock
    mock_error: MagicMock


mock_entrypoints: MockEntrypoints = MockEntrypoints(
    metadata_entry_points=MagicMock(),
    first_plugin=MagicMock(),
    second_plugin=MagicMock(),
)


class FakeDistribution(metadata.Distribution):
    """Fake metadata distribution for testing."""

    def __init__(self, fake_name: str) -> None:
        """Construct a fake distribution for testing

        :param fake_name: Fake name to return when queried
        """
        self._fake_name = fake_name

    @property
    def _normalized_name(self) -> str:
        return self._fake_name

    def locate_file(
        self, path: Union[str, PathLike[str]]
    ) -> metadata.SimplePath:  # type: ignore
        return metadata.Distribution.from_name("mlia").locate_file(path)

    def read_text(self, filename: str) -> str:
        mlia_distribution = metadata.Distribution.from_name("mlia")
        return (mlia_distribution.read_text(filename) or "").replace(
            "mlia", self._fake_name
        )


@typing.no_type_check
def _bind_entrypoint(
    entry_point: metadata.EntryPoint, distribution: metadata.Distribution
):
    """Accesses internal methods to fully setup an entrypoint."""
    # pylint: disable-next=protected-access
    return entry_point._for(distribution)


class TracebackMatcher:
    """Matcher to ensure traceback exists in log string."""

    def __eq__(self, other: object) -> bool:
        """Checks that traceback is included in a given string."""
        if isinstance(other, str):
            return "Traceback (most recent call last):\n" in other
        raise NotImplementedError


@pytest.fixture(autouse=True, name="mock_logging")
def mock_logging_fixture(monkeypatch: pytest.MonkeyPatch) -> MockLogging:
    """Mocked out logging to check values."""
    stub_logging = MockLogging(mock_debug=MagicMock(), mock_error=MagicMock())

    monkeypatch.setattr(logger, "debug", stub_logging.mock_debug)
    monkeypatch.setattr(logger, "error", stub_logging.mock_error)

    return stub_logging


@pytest.fixture(name="single_entrypoint")
def single_entrypoint_fixture(monkeypatch: pytest.MonkeyPatch) -> MockEntrypoints:
    """Create a single entrypoint to load in the plugin system."""
    global mock_entrypoints  # pylint: disable=global-statement

    entry_point = metadata.EntryPoint(
        "plugin", "tests.test_plugins:mock_entrypoints.first_plugin", "stub.entrypoint"
    )
    mlia_distribution = metadata.Distribution.from_name("mlia")
    entry_point = _bind_entrypoint(entry_point, mlia_distribution)

    mock_entrypoints = MockEntrypoints(
        metadata_entry_points=MagicMock(
            return_value=metadata.EntryPoints([entry_point])
        ),
        first_plugin=MagicMock(),
        second_plugin=MagicMock(),
    )
    mock_entrypoints.first_plugin.plugin_interface_version = "0.0.1"

    monkeypatch.setattr(
        metadata, "entry_points", mock_entrypoints.metadata_entry_points
    )

    return mock_entrypoints


@pytest.fixture(name="single_external_entrypoint")
def single_external_entrypoint_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> MockEntrypoints:
    """Create a single entrypoint for an external package to load."""
    global mock_entrypoints  # pylint: disable=global-statement

    entry_point = metadata.EntryPoint(
        "plugin", "tests.test_plugins:mock_entrypoints.first_plugin", "stub.entrypoint"
    )
    external_distribution = FakeDistribution("mlia-external-plugin")
    entry_point = _bind_entrypoint(entry_point, external_distribution)
    mock_entrypoints = MockEntrypoints(
        metadata_entry_points=MagicMock(
            return_value=metadata.EntryPoints([entry_point])
        ),
        first_plugin=MagicMock(),
        second_plugin=MagicMock(),
    )
    mock_entrypoints.first_plugin.plugin_interface_version = "0.0.1"

    monkeypatch.setattr(
        metadata, "entry_points", mock_entrypoints.metadata_entry_points
    )

    return mock_entrypoints


@pytest.fixture(name="multiple_entrypoint")
def multiple_entrypoint_fixture(monkeypatch: pytest.MonkeyPatch) -> MockEntrypoints:
    """Create multiple entrypoints for testing multiple plugins."""
    global mock_entrypoints  # pylint: disable=global-statement

    first_ep = metadata.EntryPoint(
        "first_plugin",
        "tests.test_plugins:mock_entrypoints.first_plugin",
        "stub.entrypoint",
    )
    mlia_distribution = metadata.Distribution.from_name("mlia")
    first_ep = _bind_entrypoint(first_ep, mlia_distribution)
    second_ep = metadata.EntryPoint(
        "second_plugin",
        "tests.test_plugins:mock_entrypoints.second_plugin",
        "stub.entrypoint",
    )
    external_distribution = FakeDistribution("mlia-external-plugin")
    second_ep = _bind_entrypoint(second_ep, external_distribution)

    mock_entrypoints = MockEntrypoints(
        metadata_entry_points=MagicMock(
            return_value=metadata.EntryPoints([first_ep, second_ep])
        ),
        first_plugin=MagicMock(),
        second_plugin=MagicMock(),
    )
    mock_entrypoints.first_plugin.plugin_interface_version = "0.0.1"
    mock_entrypoints.second_plugin.plugin_interface_version = "0.0.1"

    monkeypatch.setattr(
        metadata,
        "entry_points",
        mock_entrypoints.metadata_entry_points,
    )

    return mock_entrypoints


def test_plugin_loader(
    single_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Test loading a single plugin."""
    call_entry_points("stub.entrypoint", "registry_here")
    single_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call("Loading plugins from '%s'", "stub.entrypoint"),
            call(
                "Loading internal plugin '%s' from '%s'",
                "plugin",
                "tests.test_plugins:mock_entrypoints.first_plugin",
            ),
        ]
    )
    single_entrypoint.first_plugin.register.assert_called_once_with("registry_here")


def test_plugin_loader_external(
    single_external_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Test loading a plugin external to mlia."""
    call_entry_points("stub.entrypoint", "registry_here")
    single_external_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call("Loading plugins from '%s'", "stub.entrypoint"),
            call(
                "Loading external plugin '%s' from '%s' (dist '%s')",
                "plugin",
                "tests.test_plugins:mock_entrypoints.first_plugin",
                "mlia-external-plugin",
            ),
        ]
    )
    single_external_entrypoint.first_plugin.register.assert_called_once_with(
        "registry_here"
    )


def test_plugin_loader_bad_version(
    single_external_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Skip plugins that have incompatible plugin interface version."""
    single_external_entrypoint.first_plugin.plugin_interface_version = "5.0.1"

    call_entry_points("stub.entrypoint", "registry_here")
    single_external_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call("Loading plugins from '%s'", "stub.entrypoint"),
        ]
    )
    mock_logging.mock_error.assert_has_calls(
        [call("Incompatible version '%s' for plugin '%s'", "5.0.1", "plugin")]
    )
    single_external_entrypoint.first_plugin.register.assert_not_called()


def test_plugin_loader_register_fail(
    single_external_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Skip plugins that error on load."""
    single_external_entrypoint.first_plugin.register.side_effect = ValueError

    call_entry_points("stub.entrypoint", "registry_here")
    single_external_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call("Loading plugins from '%s'", "stub.entrypoint"),
        ]
    )
    mock_logging.mock_error.assert_has_calls(
        [
            call("Error loading plugin '%s'", "plugin"),
            call(TracebackMatcher()),
        ]
    )
    single_external_entrypoint.first_plugin.register.assert_called_once_with(
        "registry_here"
    )


def test_plugin_loader_many(
    multiple_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Load multiple plugins at once."""
    call_entry_points("stub.entrypoint", "registry_here")
    multiple_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call(
                "Loading internal plugin '%s' from '%s'",
                "first_plugin",
                "tests.test_plugins:mock_entrypoints.first_plugin",
            ),
            call(
                "Loading external plugin '%s' from '%s' (dist '%s')",
                "second_plugin",
                "tests.test_plugins:mock_entrypoints.second_plugin",
                "mlia-external-plugin",
            ),
        ]
    )
    multiple_entrypoint.first_plugin.register.assert_called_once_with("registry_here")
    multiple_entrypoint.second_plugin.register.assert_called_once_with("registry_here")


def test_plugin_loader_many_version_mismatch(
    multiple_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Skip version mismatch and continue loading."""
    multiple_entrypoint.first_plugin.plugin_interface_version = "5.0.1"

    call_entry_points("stub.entrypoint", "registry_here")
    multiple_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call(
                "Loading internal plugin '%s' from '%s'",
                "first_plugin",
                "tests.test_plugins:mock_entrypoints.first_plugin",
            ),
            call(
                "Loading external plugin '%s' from '%s' (dist '%s')",
                "second_plugin",
                "tests.test_plugins:mock_entrypoints.second_plugin",
                "mlia-external-plugin",
            ),
        ]
    )
    mock_logging.mock_error.assert_has_calls(
        [call("Incompatible version '%s' for plugin '%s'", "5.0.1", "first_plugin")]
    )
    multiple_entrypoint.first_plugin.register.assert_not_called()
    multiple_entrypoint.second_plugin.register.assert_called_once_with("registry_here")


def test_plugin_loader_many_error(
    multiple_entrypoint: MockEntrypoints, mock_logging: MockLogging
) -> None:
    """Skip plugin error and continue loading."""
    multiple_entrypoint.first_plugin.register.side_effect = ValueError

    call_entry_points("stub.entrypoint", "registry_here")
    multiple_entrypoint.metadata_entry_points.assert_called_once_with(
        group="stub.entrypoint"
    )
    mock_logging.mock_debug.assert_has_calls(
        [
            call(
                "Loading internal plugin '%s' from '%s'",
                "first_plugin",
                "tests.test_plugins:mock_entrypoints.first_plugin",
            ),
            call(
                "Loading external plugin '%s' from '%s' (dist '%s')",
                "second_plugin",
                "tests.test_plugins:mock_entrypoints.second_plugin",
                "mlia-external-plugin",
            ),
        ]
    )
    mock_logging.mock_error.assert_has_calls(
        [
            call("Error loading plugin '%s'", "first_plugin"),
            call(TracebackMatcher()),
        ]
    )
    multiple_entrypoint.first_plugin.register.assert_called_once_with("registry_here")
    multiple_entrypoint.second_plugin.register.assert_called_once_with("registry_here")
