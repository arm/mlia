# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for API logs_dir path handling helpers."""

from pathlib import Path

import pytest

from mlia.api import _resolve_logs_dir
from mlia.core.errors import ConfigurationError


def test_resolve_logs_dir_defaults() -> None:
    """Ensure logs_dir defaults to None when not provided."""
    assert _resolve_logs_dir(None) is None


def test_resolve_logs_dir_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure relative logs_dir resolves against CWD."""
    monkeypatch.chdir(tmp_path)
    logs_path = _resolve_logs_dir("logs")
    assert logs_path == tmp_path / "logs"
    assert logs_path.is_dir()


def test_resolve_logs_dir_file_path(tmp_path: Path) -> None:
    """Ensure logs_dir rejects file paths."""
    file_path = tmp_path / "logs"
    file_path.write_text("not a dir", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="is not a directory"):
        _resolve_logs_dir(file_path)


def test_resolve_logs_dir_mkdir_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure logs_dir raises when directory creation fails."""

    def raise_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr(Path, "mkdir", raise_error)

    with pytest.raises(ConfigurationError, match="Unable to create logs directory"):
        _resolve_logs_dir(tmp_path / "logs")


def test_resolve_logs_dir_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure logs_dir raises when not writable."""
    logs_path = tmp_path / "logs"
    logs_path.mkdir()
    monkeypatch.setattr("mlia.api.os.access", lambda *_args, **_kwargs: False)

    with pytest.raises(ConfigurationError, match="is not writable"):
        _resolve_logs_dir(logs_path)
