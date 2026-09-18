# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for typed MLIA application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer import rich_utils

import mlia.cli.settings as cli_settings
from mlia.core.errors import ConfigurationError
from mlia.core.settings import (
    ApplicationSettings,
    FilteringSettings,
    parse_filtering_settings,
)


@pytest.mark.parametrize(
    ("theme", "border", "name", "version", "highlight", "required"),
    [
        ("dark", "#B8BECB", "#6B9BF8", "#9385F8", "#F69B4F", "#ED564D"),
        ("light", "#40454F", "#0157FF", "#6A36ED", "#DB8232", "#B4181B"),
    ],
)
def test_configure_typer_help_uses_selected_theme(
    monkeypatch: pytest.MonkeyPatch,
    theme: cli_settings.ThemeName,
    border: str,
    name: str,
    version: str,
    highlight: str,
    required: str,
) -> None:
    """Typer's generated help should use the selected MLIA palette."""
    style_names = (
        "STYLE_OPTION",
        "STYLE_SWITCH",
        "STYLE_NEGATIVE_OPTION",
        "STYLE_NEGATIVE_SWITCH",
        "STYLE_METAVAR",
        "STYLE_USAGE",
        "STYLE_REQUIRED_SHORT",
        "STYLE_REQUIRED_LONG",
        "STYLE_COMMANDS_TABLE_FIRST_COLUMN",
        "STYLE_OPTIONS_PANEL_BORDER",
        "STYLE_COMMANDS_PANEL_BORDER",
    )
    for style_name in style_names:
        monkeypatch.setattr(rich_utils, style_name, "original")

    cli_settings.configure_typer_help(True, theme)

    assert rich_utils.STYLE_OPTION == f"bold {name}"
    assert rich_utils.STYLE_SWITCH == f"bold {name}"
    assert rich_utils.STYLE_NEGATIVE_OPTION == f"bold {version}"
    assert rich_utils.STYLE_NEGATIVE_SWITCH == f"bold {version}"
    assert rich_utils.STYLE_METAVAR == f"bold {version}"
    assert rich_utils.STYLE_USAGE == highlight
    assert rich_utils.STYLE_REQUIRED_SHORT == required
    assert rich_utils.STYLE_REQUIRED_LONG == required
    assert rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN == f"bold {name}"
    assert rich_utils.STYLE_OPTIONS_PANEL_BORDER == border
    assert rich_utils.STYLE_COMMANDS_PANEL_BORDER == border


def test_configure_typer_help_does_not_add_colors_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled color output should leave Typer's styles unchanged."""
    monkeypatch.setattr(rich_utils, "STYLE_OPTION", "original")

    cli_settings.configure_typer_help(False)

    assert rich_utils.STYLE_OPTION == "original"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "dark"), ("dark", "dark"), ("light", "light")],
)
def test_theme_defaults_to_dark_and_uses_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    expected: cli_settings.ThemeName,
) -> None:
    """TOML should select a theme with dark as the default."""
    monkeypatch.delenv("MLIA_LIGHT", raising=False)
    monkeypatch.delenv("MLIA_DARK", raising=False)
    config = {} if configured is None else {"theme": configured}

    assert cli_settings._theme_name(cli_settings.TOMLVerifier(config)) == expected


@pytest.mark.parametrize(
    ("environment", "configured", "expected"),
    [
        ({"MLIA_LIGHT": "1"}, "dark", "light"),
        ({"MLIA_DARK": "1"}, "light", "dark"),
        ({"MLIA_LIGHT": ""}, "dark", "dark"),
        ({"MLIA_DARK": ""}, "light", "light"),
    ],
)
def test_theme_environment_overrides_configuration(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    configured: str,
    expected: cli_settings.ThemeName,
) -> None:
    """Non-empty theme environment selectors should override TOML."""
    monkeypatch.delenv("MLIA_LIGHT", raising=False)
    monkeypatch.delenv("MLIA_DARK", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert (
        cli_settings._theme_name(cli_settings.TOMLVerifier({"theme": configured}))
        == expected
    )


def test_theme_rejects_conflicting_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contradictory environment selectors should be rejected."""
    monkeypatch.setenv("MLIA_LIGHT", "1")
    monkeypatch.setenv("MLIA_DARK", "1")

    with pytest.raises(ConfigurationError, match="cannot both be set"):
        cli_settings._theme_name(cli_settings.TOMLVerifier())


@pytest.mark.parametrize("configured", ["blue", "LIGHT", 1])
def test_theme_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, configured: object
) -> None:
    """Theme configuration should accept only lowercase supported names."""
    monkeypatch.delenv("MLIA_LIGHT", raising=False)
    monkeypatch.delenv("MLIA_DARK", raising=False)

    with pytest.raises(ConfigurationError, match='value "theme"'):
        cli_settings._theme_name(cli_settings.TOMLVerifier({"theme": configured}))


def test_new_settings_preserves_source_theme(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuilding command settings should preserve the selected theme."""
    monkeypatch.delenv("MLIA_LIGHT", raising=False)
    monkeypatch.delenv("MLIA_DARK", raising=False)
    monkeypatch.setattr(cli_settings, "_color_enabled", lambda: None)
    source = ApplicationSettings(theme="light")

    settings = cli_settings.new_settings(source=source)

    assert settings.theme == "light"
    assert str(settings.console.get_style("tbl.border")) == "#40454f"


def test_new_settings_retains_theme_when_color_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Color suppression should not discard the selected theme name."""
    monkeypatch.delenv("MLIA_LIGHT", raising=False)
    monkeypatch.delenv("MLIA_DARK", raising=False)
    monkeypatch.setattr(cli_settings, "_color_enabled", lambda: None)
    source = ApplicationSettings(color=False, theme="light")

    settings = cli_settings.new_settings(source=source)

    assert settings.color is False
    assert settings.theme == "light"
    assert str(settings.console.get_style("tbl.version")) == "none"


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
