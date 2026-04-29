# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the copyright header pre-commit hook."""

import datetime
import importlib.util
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def load_checker_module() -> ModuleType:
    """Load the copyright header checker module from disk."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "pre_commit_hooks"
        / "check_copyright_header.py"
    )
    spec = importlib.util.spec_from_file_location("check_copyright_header", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="checker_module")
def fixture_checker_module() -> ModuleType:
    """Load the copyright header checker module."""
    return load_checker_module()


def test_markdown_sidecar_header_takes_precedence(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Test markdown files can use a sidecar license header."""
    current_year = datetime.datetime.now().year
    markdown_file = tmp_path / "generated.md"
    markdown_file.write_text(
        "# Generated content\nNo inline header.\n", encoding="utf-8"
    )
    (tmp_path / "generated.md.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(markdown_file)]) is True


def test_yaml_sidecar_header_is_accepted(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Test YAML files can use a sidecar license header."""
    current_year = datetime.datetime.now().year
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("schema: spec-driven\n", encoding="utf-8")
    (tmp_path / "config.yaml.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(yaml_file)]) is True


def test_markdown_inline_header_still_works_without_sidecar(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Test markdown files still support inline SPDX headers."""
    current_year = datetime.datetime.now().year
    markdown_file = tmp_path / "inline.md"
    markdown_file.write_text(
        "\n"
        f"<!-- SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited -->\n"
        "Body\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(markdown_file)]) is True


def test_json_file_still_requires_sidecar(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test JSON files still require a sidecar license file."""
    json_file = tmp_path / "schema.json"
    json_file.write_text('{"schema": true}\n', encoding="utf-8")

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(json_file)]) is False
    captured = capsys.readouterr()
    assert "requires a sidecar" in captured.out


def test_markdown_conflicting_inline_and_sidecar_headers_fail(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test conflicting inline and sidecar copyright headers fail."""
    current_year = datetime.datetime.now().year
    markdown_file = tmp_path / "generated.md"
    markdown_file.write_text(
        "\n"
        "<!-- SPDX-FileCopyrightText: Copyright 2025, Someone Else -->\n"
        "Generated content\n",
        encoding="utf-8",
    )
    (tmp_path / "generated.md.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(markdown_file)]) is False
    captured = capsys.readouterr()
    assert "conflict" in captured.out


def test_json_sidecar_only_formats_skip_conflict_check(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Test sidecar-only formats do not cross-check content for conflicts."""
    current_year = datetime.datetime.now().year
    json_file = tmp_path / "schema.json"
    json_file.write_text(
        '{"note":"SPDX-FileCopyrightText: Copyright 2025, Someone Else"}\n',
        encoding="utf-8",
    )
    (tmp_path / "schema.json.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(json_file)]) is True


def test_matching_inline_and_sidecar_headers_pass(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    """Test matching inline and sidecar headers are accepted."""
    current_year = datetime.datetime.now().year
    markdown_file = tmp_path / "matching.md"
    header_line = (
        f"<!-- SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited -->"
    )
    markdown_file.write_text(f"\n{header_line}\nBody\n", encoding="utf-8")
    (tmp_path / "matching.md.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(markdown_file)]) is True


def test_missing_files_are_skipped(checker_module: ModuleType) -> None:
    """Test missing files are ignored."""
    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header(["missing-file.txt"]) is True


def test_license_text_files_are_skipped(
    tmp_path: Path,
    checker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test canonical license texts are ignored by the copyright checker."""
    license_file = tmp_path / "LICENSES" / "BSD-3-Clause.txt"
    license_file.parent.mkdir()
    license_file.write_text(
        "Copyright <YEAR> <COPYRIGHT HOLDER>\n"
        "Redistribution and use in source and binary forms...\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header(["LICENSES/BSD-3-Clause.txt"])


def test_license_files_do_not_probe_for_nested_sidecars(
    checker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test sidecar files are treated as terminal inputs."""
    fake_exists = MagicMock(return_value=False)
    monkeypatch.setattr(checker_module.os.path, "exists", fake_exists)

    assert checker_module.resolve_header_source("generated.md.license") == (
        "generated.md.license",
        "",
    )
    fake_exists.assert_not_called()


def test_unreadable_file_during_conflict_check_is_skipped(
    tmp_path: Path,
    checker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test unreadable files warn and are skipped during conflict checks."""
    current_year = datetime.datetime.now().year
    markdown_file = tmp_path / "generated.md"
    markdown_file.write_text(
        "\n"
        f"<!-- SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited -->\n"
        "Body\n",
        encoding="utf-8",
    )
    (tmp_path / "generated.md.license").write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checker_module,
        "open",
        MagicMock(
            side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        ),
        raising=False,
    )
    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(markdown_file)]) is True
    captured = capsys.readouterr()
    assert "WARN: Cannot check" in captured.out


def test_unreadable_header_file_is_skipped(
    tmp_path: Path,
    checker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test unreadable header files warn and are skipped."""
    current_year = datetime.datetime.now().year
    yaml_file = tmp_path / "config.yaml"
    header_file = tmp_path / "config.yaml.license"
    yaml_file.write_text("schema: spec-driven\n", encoding="utf-8")
    header_file.write_text(
        f"SPDX-FileCopyrightText: Copyright {current_year}, Arm Limited\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checker_module,
        "open",
        MagicMock(
            side_effect=[
                StringIO("schema: spec-driven\n"),
                PermissionError("permission denied"),
            ]
        ),
        raising=False,
    )
    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(yaml_file)]) is True
    captured = capsys.readouterr()
    assert "WARN: Cannot check" in captured.out


def test_outdated_inline_header_fails(
    tmp_path: Path,
    checker_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test outdated inline headers fail the check."""
    markdown_file = tmp_path / "outdated.md"
    markdown_file.write_text(
        "\n<!-- SPDX-FileCopyrightText: Copyright 2025, Arm Limited -->\nBody\n",
        encoding="utf-8",
    )

    checker = checker_module.CopyrightHeaderChecker()

    assert checker.check_files_have_updated_header([str(markdown_file)]) is False
    captured = capsys.readouterr()
    assert "out of date" in captured.out
