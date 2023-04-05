# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for misc util functions."""
from subprocess import CalledProcessError  # nosec
from unittest.mock import MagicMock

import pytest

from mlia.utils.misc import get_pkg_version
from mlia.utils.misc import is_docker_available
from mlia.utils.misc import yes


@pytest.mark.parametrize(
    "response, expected_result",
    [
        ["Y", True],
        ["y", True],
        ["N", False],
        ["n", False],
    ],
)
def test_yes(
    monkeypatch: pytest.MonkeyPatch, expected_result: bool, response: str
) -> None:
    """Test yes function."""
    monkeypatch.setattr("builtins.input", MagicMock(return_value=response))
    assert yes("some_prompt") == expected_result


@pytest.mark.parametrize("response", ["some version", FileNotFoundError()])
def test_get_pkg_version(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    """Test get_tosa_version."""
    monkeypatch.setattr("importlib.metadata.version", MagicMock(return_value=response))
    assert get_pkg_version("any name") == response


@pytest.mark.parametrize(
    ("mock_run", "expected_result"),
    (
        (MagicMock(), True),
        (
            MagicMock(
                side_effect=CalledProcessError(12, "TEST"),
            ),
            False,
        ),
    ),
)
def test_is_docker_available(
    monkeypatch: pytest.MonkeyPatch, mock_run: MagicMock, expected_result: bool
) -> None:
    """Test function is_docker_available()."""
    monkeypatch.setattr("mlia.utils.misc.run", mock_run)
    assert is_docker_available() == expected_result
