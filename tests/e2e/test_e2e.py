# Copyright 2021, Arm Ltd.
"""End to end tests for MLIA CLI."""
import os
from pathlib import Path
from typing import List
from typing import Optional

import pytest
from mlia.utils.proc import CommandExecutor


def run_command(cmd: List[str]) -> None:
    """Run command."""
    print(f"Run command: {' '.join(cmd)}")

    executor = CommandExecutor()
    running_command = executor.submit(cmd)
    running_command.wait(redirect_output=True)

    exit_code = running_command.exit_code()
    if exit_code is None or exit_code != 0:
        raise Exception(f"Execution failed {' '.join(cmd)}")


def install_aiet_artifacts(system_dirs: List[Path], software_dirs: List[Path]) -> None:
    """Install AIET software and systems."""
    run_command(["aiet", "--version"])

    for system in system_dirs:
        if not system.is_dir():
            raise Exception(f"Wrong directory {system}. Unable to install system")

        run_command(["aiet", "system", "install", "-s", str(system)])
        print(f"System {system} installed")

    run_command(["aiet", "system", "list"])

    for software in software_dirs:
        if not software.is_dir():
            raise Exception(f"Wrong directory {software}. Unable to install software")

        run_command(["aiet", "software", "install", "-s", str(software)])
        print(f"Software {software} installed")

    run_command(["aiet", "software", "list"])


def get_config_dir() -> Optional[Path]:
    """Get configuration directory path."""
    config_dir = os.environ.get("MLIA_E2E_CONFIG")
    if not config_dir:
        return None

    config_dir_path = Path(config_dir)
    if not config_dir_path.is_dir():
        raise Exception(f"Wrong config directory {config_dir}")

    return config_dir_path


def get_directories(parent_dir: Path) -> List[Path]:
    """Get all directory inside parent directory."""
    if not parent_dir.is_dir():
        return []

    return [item for item in parent_dir.iterdir() if item.is_dir()]


def discover_and_install_aiet_artifacts() -> None:
    """Discover and install AIET artifacts."""
    config_dir_path = get_config_dir()
    if not config_dir_path:
        return

    print(f"Found e2e config directory {config_dir_path}")
    systems_dirs = get_directories(config_dir_path / "systems")
    software_dirs = get_directories(config_dir_path / "software")

    install_aiet_artifacts(systems_dirs, software_dirs)


def get_models() -> List[Path]:
    """Get list of pathes to the models to test."""
    config_dir_path = get_config_dir()
    if not config_dir_path:
        return []

    return [Path("tests/test_resources/models/simple_3_layers_model.tflite")]


@pytest.mark.e2e
class TestEndToEnd:
    """End to end tests."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up test class."""
        discover_and_install_aiet_artifacts()

    @pytest.mark.parametrize("model", get_models())
    @pytest.mark.parametrize("device", ["ethos-u55", "ethos-u65"])
    @pytest.mark.parametrize("fmt", ["json", "txt", "csv"])
    def test_operators(self, model: Path, device: str, fmt: str) -> None:
        """Test command 'operators'."""
        command = [
            "mlia",
            "operators",
            "--device",
            device,
            "--output-format",
            fmt,
            str(model),
        ]

        run_command(command)

    @pytest.mark.parametrize("model", get_models())
    @pytest.mark.parametrize("device", ["ethos-u55", "ethos-u65"])
    def test_performance(
        self,
        model: Path,
        device: str,
    ) -> None:
        """Test command 'performance'."""
        command = ["mlia", "performance", "--verbose", "--device", device, str(model)]

        run_command(command)
