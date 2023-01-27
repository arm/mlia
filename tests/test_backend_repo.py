# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend repository."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlia.backend.repo import BackendRepository
from mlia.backend.repo import get_backend_repository


def test_get_backend_repository(tmp_path: Path) -> None:
    """Test function get_backend_repository."""
    repo_path = tmp_path / "repo"
    repo = get_backend_repository(repo_path)

    assert isinstance(repo, BackendRepository)

    backends_dir = repo_path / "backends"
    assert backends_dir.is_dir()
    assert not list(backends_dir.iterdir())

    config_file = repo_path / "mlia_config.json"
    assert config_file.is_file()
    assert json.loads(config_file.read_text()) == {"backends": []}


def test_backend_repository_wrong_directory(tmp_path: Path) -> None:
    """Test that repository instance should throw error for the wrong directory."""
    with pytest.raises(
        Exception, match=f"Directory {tmp_path} could not be used as MLIA repository."
    ):
        BackendRepository(tmp_path)


def test_empty_backend_repository(tmp_path: Path) -> None:
    """Test empty backend repository."""
    repo_path = tmp_path / "repo"
    repo = get_backend_repository(repo_path)

    assert not repo.is_backend_installed("sample_backend")

    with pytest.raises(Exception, match="Backend sample_backend is not installed."):
        repo.remove_backend("sample_backend")

    with pytest.raises(Exception, match="Backend sample_backend is not installed."):
        repo.get_backend_settings("sample_backend")


def test_adding_backend(tmp_path: Path) -> None:
    """Test adding backend to the repository."""
    repo_path = tmp_path / "repo"
    repo = get_backend_repository(repo_path)

    backend_path = tmp_path.joinpath("backend")
    backend_path.mkdir()

    settings = {"param": "value"}
    repo.add_backend("sample_backend", backend_path, settings)

    backends_dir = repo_path / "backends"
    assert backends_dir.is_dir()
    assert not list(backends_dir.iterdir())

    assert repo.is_backend_installed("sample_backend")
    expected_settings = {
        "param": "value",
        "backend_path": backend_path.as_posix(),
    }
    assert repo.get_backend_settings("sample_backend") == (
        backend_path,
        expected_settings,
    )

    config_file = repo_path / "mlia_config.json"
    expected_content = {
        "backends": [
            {
                "name": "sample_backend",
                "settings": {
                    "backend_path": backend_path.as_posix(),
                    "param": "value",
                },
            }
        ]
    }
    assert json.loads(config_file.read_text()) == expected_content

    with pytest.raises(Exception, match="Backend sample_backend already installed"):
        repo.add_backend("sample_backend", backend_path, settings)

    repo.remove_backend("sample_backend")
    assert not repo.is_backend_installed("sample_backend")


def test_copy_backend(tmp_path: Path) -> None:
    """Test copying backend to the repository."""
    repo_path = tmp_path / "repo"
    repo = get_backend_repository(repo_path)

    backend_path = tmp_path.joinpath("backend")
    backend_path.mkdir()

    backend_path.joinpath("sample.txt").touch()

    settings = {"param": "value"}
    repo.copy_backend("sample_backend", backend_path, "sample_backend_dir", settings)

    repo_backend_path = repo_path.joinpath("backends", "sample_backend_dir")
    assert repo_backend_path.is_dir()
    assert repo_backend_path.joinpath("sample.txt").is_file()

    config_file = repo_path / "mlia_config.json"
    expected_content = {
        "backends": [
            {
                "name": "sample_backend",
                "settings": {
                    "backend_dir": "sample_backend_dir",
                    "param": "value",
                },
            }
        ]
    }
    assert json.loads(config_file.read_text()) == expected_content

    expected_settings = {
        "param": "value",
        "backend_dir": "sample_backend_dir",
    }
    assert repo.get_backend_settings("sample_backend") == (
        repo_backend_path,
        expected_settings,
    )

    repo.remove_backend("sample_backend")
    assert not repo_backend_path.exists()
