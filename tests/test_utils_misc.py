# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for misc util functions."""

from unittest.mock import MagicMock

import pytest

from mlia.utils.misc import MetadataError, get_pkg_version, merge, summarize_list, yes


def test_summarize_list() -> None:
    """Bound list summaries and report omitted item counts."""
    assert summarize_list([]) == ""
    assert summarize_list(["a", "b"]) == "a, b"
    assert summarize_list(range(7)) == "0, 1, 2, 3, 4, ... (2 more omitted)"
    assert summarize_list(range(4), limit=2, separator="; ") == (
        "0; 1; ... (2 more omitted)"
    )


def test_summarize_list_rejects_nonpositive_limit() -> None:
    """A summary must always permit at least one visible item."""
    with pytest.raises(ValueError, match="must be positive"):
        summarize_list(["item"], limit=0)


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


def test_recursive_dictionary_merge() -> None:
    a = {"A": {"B": 1, "C": 2}}
    b = {"A": {"B": 8}}

    assert merge(a, b) == {"A": {"B": 8, "C": 2}}
