# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for typed MLIA application settings."""

from __future__ import annotations

from pathlib import Path

import pytest

import mlia.cli.settings as cli_settings
from mlia.core.errors import ConfigurationError
from mlia.core.settings import (
    ApplicationSettings,
    FilteringSettings,
    parse_filtering_settings,
)


def test_filtering_defaults_are_empty() -> None:
    settings = ApplicationSettings()

    assert settings.filtering.collapse == ()


def test_upstream_config_loader_parses_filtering_and_plugin_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[filtering]
collapse = []

[plugins.example]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_settings, "_CONFIG_PATH", config)
    monkeypatch.setattr(cli_settings, "_color_enabled", lambda: False)

    settings = cli_settings.new_settings()

    assert settings.filtering == FilteringSettings(collapse=())
    assert settings.for_plugin("example") == {"enabled": True}


def test_explicit_structured_collapse_rules() -> None:
    settings = parse_filtering_settings(
        {
            "collapse": [
                {
                    "kind": "arbitrary",
                    "attribute": "source",
                    "globs": ["first/*", "second/*"],
                },
                {"kind": "other", "attribute": "label", "globs": ["match"]},
            ]
        }
    )

    assert [(rule.kind, rule.attribute, rule.globs) for rule in settings.collapse] == [
        ("arbitrary", "source", ("first/*", "second/*")),
        ("other", "label", ("match",)),
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"unknown": []}, "Unknown filtering"),
        ({"collapse": 1}, "collapse.*array of tables"),
        ({"collapse": [1]}, "collapse\\[0\\].*table"),
        ({"collapse": [{"unknown": 1}]}, "Unknown filtering.collapse"),
        (
            {"collapse": [{"kind": "", "attribute": "a", "globs": ["x"]}]},
            "kind.*non-empty string",
        ),
        (
            {"collapse": [{"kind": "k", "attribute": "", "globs": ["x"]}]},
            "attribute.*non-empty string",
        ),
        (
            {"collapse": [{"kind": "k", "attribute": "a", "globs": []}]},
            "globs.*non-empty array",
        ),
    ],
)
def test_malformed_filtering_configuration_is_rejected(
    value: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        parse_filtering_settings(value)
