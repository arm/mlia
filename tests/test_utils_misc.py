# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for misc util functions."""
import copy
from subprocess import CalledProcessError  # nosec
from unittest.mock import MagicMock

import pytest

from mlia.utils.misc import dict_to_list
from mlia.utils.misc import get_pkg_version
from mlia.utils.misc import is_docker_available
from mlia.utils.misc import list_to_dict
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


def test_list_to_dict() -> None:
    """Test convert list of dicts to dict."""
    test_case_1 = [{"foo": "bar", "this": "is"}, {"foo": "a", "test": "case", 1: 2}]
    test_case_2 = [{"foo": "bar", "this": "is"}, {"a": "test"}]
    test_case_3 = copy.deepcopy(test_case_1)

    expected_case_1 = {"bar": {"this": "is"}, "a": {"test": "case", 1: 2}}
    assert expected_case_1 == list_to_dict(test_case_1, "foo")

    with pytest.raises(KeyError):
        list_to_dict(test_case_2, "foo")

    assert dict_to_list(list_to_dict(test_case_3, "foo"), "foo") == [  # type: ignore
        {"foo": "bar", "this": "is"},
        {"foo": "a", "test": "case", 1: 2},
    ]


def test_dict_to_list() -> None:
    """Test convert dict to list of dicts."""
    test_case = {"bar": {"this": "is"}, "a": {"test": "case", 1: 2}}
    expected_test_case = [
        {"foo": "bar", "this": "is"},
        {"foo": "a", "test": "case", 1: 2},
    ]
    assert expected_test_case == dict_to_list(test_case, "foo")
    assert list_to_dict(dict_to_list(test_case, "foo"), "foo") == test_case
