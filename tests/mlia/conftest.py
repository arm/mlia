# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest conf module."""
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pytest

from mlia.core.context import ExecutionContext


@pytest.fixture(scope="session", name="test_resources_path")
def fixture_test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture(name="dummy_context")
def fixture_dummy_context(tmpdir: str) -> ExecutionContext:
    """Return dummy context fixture."""
    return ExecutionContext(working_dir=tmpdir)


@pytest.fixture(scope="session")
def test_systems_path(test_resources_path: Path) -> Path:
    """Return test systems path in a pytest fixture."""
    return test_resources_path / "backends" / "systems"


@pytest.fixture(scope="session")
def test_applications_path(test_resources_path: Path) -> Path:
    """Return test applications path in a pytest fixture."""
    return test_resources_path / "backends" / "applications"


@pytest.fixture(scope="session")
def non_optimised_input_model_file(test_tflite_model: Path) -> Path:
    """Provide the path to a quantized dummy model file."""
    return test_tflite_model


@pytest.fixture(scope="session")
def optimised_input_model_file(test_tflite_vela_model: Path) -> Path:
    """Provide path to Vela-optimised dummy model file."""
    return test_tflite_vela_model


@pytest.fixture(scope="session")
def invalid_input_model_file(test_tflite_invalid_model: Path) -> Path:
    """Provide the path to an invalid dummy model file."""
    return test_tflite_invalid_model


@pytest.fixture(autouse=True)
def test_resources(monkeypatch: pytest.MonkeyPatch, test_resources_path: Path) -> Any:
    """Force using test resources as middleware's repository."""

    def get_test_resources() -> Path:
        """Return path to the test resources."""
        return test_resources_path / "backends"

    monkeypatch.setattr("mlia.backend.fs.get_backend_resources", get_test_resources)
    yield


def create_archive(
    archive_name: str, source: Path, destination: Path, with_root_folder: bool = False
) -> None:
    """Create archive from directory source."""
    with tarfile.open(destination / archive_name, mode="w:gz") as tar:
        for item in source.iterdir():
            item_name = item.name
            if with_root_folder:
                item_name = f"{source.name}/{item_name}"
            tar.add(item, item_name)


def process_directory(source: Path, destination: Path) -> None:
    """Process resource directory."""
    destination.mkdir()

    for item in source.iterdir():
        if item.is_dir():
            create_archive(f"{item.name}.tar.gz", item, destination)
            create_archive(f"{item.name}_dir.tar.gz", item, destination, True)


@pytest.fixture(scope="session", autouse=True)
def add_archives(
    test_resources_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """Generate archives of the test resources."""
    tmp_path = tmp_path_factory.mktemp("archives")

    archives_path = tmp_path / "archives"
    archives_path.mkdir()

    if (archives_path_link := test_resources_path / "archives").is_symlink():
        archives_path_link.unlink()

    archives_path_link.symlink_to(archives_path, target_is_directory=True)

    for item in ["applications", "systems"]:
        process_directory(test_resources_path / "backends" / item, archives_path / item)

    yield

    archives_path_link.unlink()
    shutil.rmtree(tmp_path)
