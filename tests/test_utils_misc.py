# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for misc util functions."""
from unittest.mock import MagicMock

import pytest

from mlia.utils.misc import get_pkg_version
from mlia.utils.misc import MetadataError
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


def test_get_pkg_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_pkg_version."""
    response = "some version"
    monkeypatch.setattr("importlib.metadata.version", MagicMock(return_value=response))
    assert get_pkg_version("any name") == response


def test_get_pkg_version_metadata_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_pkg_version throwa MetdataError error."""
    exc_file_not_found = FileNotFoundError()
    monkeypatch.setattr(
        "importlib.metadata.version", MagicMock(side_effect=exc_file_not_found)
    )
    with pytest.raises(MetadataError):
        get_pkg_version("any name")
