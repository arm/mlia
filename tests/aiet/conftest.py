# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-outer-name
"""conftest for pytest."""
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from aiet.backend.common import get_backend_configs


@pytest.fixture(scope="session")
def test_systems_path(test_resources_path: Path) -> Path:
    """Return test systems path in a pytest fixture."""
    return test_resources_path / "systems"


@pytest.fixture(scope="session")
def test_applications_path(test_resources_path: Path) -> Path:
    """Return test applications path in a pytest fixture."""
    return test_resources_path / "applications"


@pytest.fixture(scope="session")
def test_tools_path(test_resources_path: Path) -> Path:
    """Return test tools path in a pytest fixture."""
    return test_resources_path / "tools"


@pytest.fixture(scope="session")
def test_resources_path() -> Path:
    """Return test resources path in a pytest fixture."""
    current_path = Path(__file__).parent.absolute()
    return current_path / "test_resources"


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
        return test_resources_path

    monkeypatch.setattr("aiet.utils.fs.get_aiet_resources", get_test_resources)
    yield


@pytest.fixture(scope="session", autouse=True)
def add_tools(test_resources_path: Path) -> Any:
    """Symlink the tools from the original resources path to the test resources path."""
    # tool_dirs = get_available_tool_directory_names()
    tool_dirs = [cfg.parent for cfg in get_backend_configs("tools")]

    links = {
        src_dir: (test_resources_path / "tools" / src_dir.name) for src_dir in tool_dirs
    }
    for src_dir, dst_dir in links.items():
        if not dst_dir.exists():
            dst_dir.symlink_to(src_dir, target_is_directory=True)
    yield
    # Remove symlinks
    for dst_dir in links.values():
        if dst_dir.is_symlink():
            dst_dir.unlink()


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
        archives_path.unlink()

    archives_path_link.symlink_to(archives_path, target_is_directory=True)

    for item in ["applications", "systems"]:
        process_directory(test_resources_path / item, archives_path / item)

    yield

    archives_path_link.unlink()
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="module")
def cli_runner() -> CliRunner:
    """Return CliRunner instance in a pytest fixture."""
    return CliRunner()
