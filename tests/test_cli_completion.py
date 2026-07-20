# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for lightweight CLI completion inventories."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from typer.testing import CliRunner

import mlia
from mlia.cli import commands as cli_commands
from mlia.cli import completion
from mlia.cli import main as cli_main


def _add_target_profile(namespace_dir: Path, profile_name: str) -> None:
    """Create a target profile under a fake MLIA namespace directory."""
    profiles_dir = namespace_dir / "resources" / "target_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / f"{profile_name}.toml").write_text(
        f'profile_name = "{profile_name}"\n',
        encoding="utf-8",
    )


def test_target_profile_names_scan_namespace_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profile names should be sorted and deduplicated across namespace paths."""
    namespace_a = tmp_path / "package_a" / "mlia"
    namespace_b = tmp_path / "package_b" / "mlia"
    _add_target_profile(namespace_a, "target-b")
    _add_target_profile(namespace_a, "target-a")
    _add_target_profile(namespace_b, "target-a")
    _add_target_profile(namespace_b, "target-c")
    (namespace_b / "resources" / "target_profiles" / "README.md").write_text(
        "ignored",
        encoding="utf-8",
    )
    monkeypatch.setattr(mlia, "__path__", [str(namespace_a), str(namespace_b)])

    assert completion.target_profile_names() == ("target-a", "target-b", "target-c")


def test_target_profile_names_skip_missing_namespace_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing resource directories should produce an empty inventory."""
    monkeypatch.setattr(mlia, "__path__", [str(tmp_path / "missing")])

    assert completion.target_profile_names() == ()


def test_root_completion_lists_static_commands_with_backend_option_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root completion should build the full plugin-provided command tree."""
    discover_backend_option_specs = MagicMock(return_value=[])
    monkeypatch.setattr(
        cli_commands, "discover_backend_option_specs", discover_backend_option_specs
    )
    monkeypatch.setenv("_MLIA_COMPLETE", "complete_bash")

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [],
        prog_name="mlia",
        env={
            "_MLIA_COMPLETE": "complete_bash",
            "COMP_WORDS": "mlia ",
            "COMP_CWORD": "1",
        },
    )

    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["check", "backend", "target"]
    discover_backend_option_specs.assert_called_once_with()


def test_check_option_completion_includes_plugin_provided_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Option completion should include plugin-derived backend options."""
    discover_backend_option_specs = MagicMock(
        return_value=[
            {
                "module": "backend_a",
                "backend": "backend-a",
                "config_key": "optimization_level",
                "click_option": click.Option(
                    [
                        "--backend-a.optimization-level",
                        "backend_a_optimization_level",
                    ],
                    type=click.Choice(["0", "1", "2"]),
                    help="Set optimization level.",
                ),
            }
        ]
    )
    monkeypatch.setattr(
        cli_commands, "discover_backend_option_specs", discover_backend_option_specs
    )
    monkeypatch.setenv("_MLIA_COMPLETE", "complete_bash")

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [],
        prog_name="mlia",
        env={
            "_MLIA_COMPLETE": "complete_bash",
            "COMP_WORDS": "mlia check --",
            "COMP_CWORD": "2",
        },
    )

    assert result.exit_code == 0
    assert "--target-profile" in result.stdout.splitlines()
    assert "--backend" in result.stdout.splitlines()
    assert "--backend-a.optimization-level" in result.stdout.splitlines()
    discover_backend_option_specs.assert_called_once_with()


@pytest.mark.parametrize(
    ("comp_words", "comp_cword"),
    [
        ("mlia check model.tflite --target-profile profile --backend x", "6"),
        ("mlia backend install x", "3"),
        ("mlia backend uninstall x", "3"),
    ],
)
def test_backend_bash_completion_returns_no_sentinel_without_matches(
    comp_words: str,
    comp_cword: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bash completion should return no backend sentinel candidate."""
    monkeypatch.setattr(cli_commands, "get_selectable_backends", lambda: [])
    monkeypatch.setenv("_MLIA_COMPLETE", "complete_bash")

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [],
        prog_name="mlia",
        env={
            "_MLIA_COMPLETE": "complete_bash",
            "COMP_WORDS": comp_words,
            "COMP_CWORD": comp_cword,
        },
    )

    assert result.exit_code == 0
    assert result.stdout == "\n"


@pytest.mark.parametrize(
    "completion_args",
    [
        "mlia check model.tflite --target-profile profile --backend x",
        "mlia backend install x",
        "mlia backend uninstall x",
    ],
)
def test_backend_zsh_completion_preserves_file_fallback(
    completion_args: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zsh completion should fall back to files without backend matches."""
    monkeypatch.setattr(cli_commands, "get_selectable_backends", lambda: [])
    monkeypatch.setenv("_MLIA_COMPLETE", "complete_zsh")

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [],
        prog_name="mlia",
        env={
            "_MLIA_COMPLETE": "complete_zsh",
            "_TYPER_COMPLETE_ARGS": completion_args,
        },
    )

    assert result.exit_code == 0
    assert result.stdout == "_files\n"


@pytest.mark.parametrize(
    ("completion_mode", "environment", "expected_output"),
    [
        (
            "complete_bash",
            {
                "COMP_WORDS": "mlia check model.tflite --target-profile x",
                "COMP_CWORD": "4",
            },
            "\n",
        ),
        (
            "complete_zsh",
            {"_TYPER_COMPLETE_ARGS": ("mlia check model.tflite --target-profile x")},
            "_files\n",
        ),
    ],
)
def test_target_profile_completion_preserves_shell_fallback(
    completion_mode: str,
    environment: dict[str, str],
    expected_output: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No packaged profile match should preserve the shell's fallback."""
    monkeypatch.setattr(cli_commands, "target_profile_names", lambda: ())
    monkeypatch.setenv("_MLIA_COMPLETE", completion_mode)

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [],
        prog_name="mlia",
        env={"_MLIA_COMPLETE": completion_mode, **environment},
    )

    assert result.exit_code == 0
    assert result.stdout == expected_output
