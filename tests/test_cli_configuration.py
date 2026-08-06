# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI backend auto-install command behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

import mlia.cli.settings as cli_settings


@pytest.mark.parametrize(
    "environment, result",
    [
        ({}, None),
        ({"NO_COLOR": "1"}, False),
        ({"NO_COLOR": ""}, None),
        ({"MLIA_NO_COLOR": "1"}, False),
        ({"MLIA_NO_COLOR": ""}, None),
        ({"NO_COLOR": "", "MLIA_NO_COLOR": ""}, None),
        ({"NO_COLOR": "1", "MLIA_NO_COLOR": ""}, False),
        ({"NO_COLOR": "", "MLIA_NO_COLOR": "1"}, False),
        ({"NO_COLOR": "1", "MLIA_NO_COLOR": "1"}, False),
        ({"TERM": ""}, None),
        ({"TERM": "dumb"}, False),
    ],
)
def test_color_enabled_responds_to_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    result: bool,
) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli_settings.tomllib, "load", lambda x: {})

    with mock.patch.dict("os.environ", clear=True, **environment):
        assert cli_settings._color_enabled() is result


@pytest.mark.parametrize(
    "keyword",
    [
        "colour",
        "use_color",
    ],
)
@mock.patch("pathlib.Path.exists", lambda a: True)
def test_configuration_rejects_bad_keywords(
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
) -> None:
    warning_mock = MagicMock()

    monkeypatch.setattr(cli_settings.tomllib, "load", lambda x: {keyword: ""})
    monkeypatch.setattr(cli_settings.logger, "warning", warning_mock)

    with mock.patch.object(Path, "open"):
        cli_settings._read_config()

    warning_mock.assert_called_once_with(
        "Unknown top-level MLIA configuration key: %s", keyword
    )
