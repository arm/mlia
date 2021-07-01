# Copyright 2021, Arm Ltd.
"""End to end tests for MLIA CLI."""
import os
import shutil
import subprocess
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import pytest
from mlia.utils.proc import CommandExecutor
from mlia.utils.proc import working_directory


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


def get_tflite_models() -> List[Path]:
    """Get list of paths to the tflite models to test."""
    config_dir_path = get_config_dir()
    if not config_dir_path:
        return []

    return [Path("tests/test_resources/models/simple_3_layers_model.tflite")]


def get_keras_models() -> List[Path]:
    """Get list of paths to the keras models to test."""
    config_dir_path = get_config_dir()
    if not config_dir_path:
        return []

    return [Path("tests/test_resources/models/simple_model.h5")]


@pytest.mark.e2e
class TestEndToEnd:
    """End to end tests."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up test class."""
        discover_and_install_aiet_artifacts()

    @pytest.mark.parametrize("model", get_tflite_models())
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

    @pytest.mark.parametrize("model", get_tflite_models())
    @pytest.mark.parametrize("device", ["ethos-u55", "ethos-u65"])
    def test_performance(
        self,
        model: Path,
        device: str,
    ) -> None:
        """Test command 'performance'."""
        command = ["mlia", "performance", "--device", device, str(model)]

        run_command(command)

    @pytest.mark.parametrize("model", get_keras_models())
    @pytest.mark.parametrize("device", ["ethos-u55", "ethos-u65"])
    @pytest.mark.parametrize("optimization", [("pruning", "0.5"), ("clustering", 32)])
    def test_estimate_optimized_performance(
        self, model: Path, device: str, optimization: Tuple[str, str]
    ) -> None:
        """Test command 'estimate_optimized_performance'."""
        command = [
            "mlia",
            "estimate_optimized_performance",
            str(model),
            "--device",
            device,
            "--optimization-type",
            optimization[0],
            "--optimization-target",
            optimization[1],
        ]

        run_command(command)

    def test_installation_script(self, tmp_path: Path) -> None:
        """Test MLIA installation script."""
        config_dir = get_config_dir()
        if not config_dir:
            raise Exception("E2E configuration directory is not provided")

        install_dir_path = tmp_path / "dist"
        install_dir_path.mkdir()

        def copy_archives(subfolder_path: Path, pattern: str = "*.tar.gz") -> None:
            """Copy archives into installation dir."""
            if not subfolder_path.is_dir():
                return

            archives = subfolder_path.glob(pattern)
            for archive in archives:
                shutil.copy2(archive, install_dir_path)

        for item in ["systems", "software"]:
            copy_archives(config_dir / item)

        def copy_env_path(env_var: str) -> None:
            """Copy file to the installation dir."""
            env_var_value = os.environ.get(env_var)

            if env_var_value:
                env_var_path = Path(env_var_value)
                shutil.copy2(env_var_path, install_dir_path)

        for item in ["AIET_ARTIFACT_PATH", "MLIA_ARTIFACT_PATH"]:
            copy_env_path(item)

        shutil.copy2("scripts/install.sh", tmp_path)

        with working_directory(tmp_path):
            run_command(["ls", "-R", str(tmp_path)])

            venv = "e2e_venv"
            command = ["./install.sh", "-d", str(install_dir_path), "-e", venv]
            run_command(command)

            assert Path(venv).is_dir()

            def run_in_env(cmd: str) -> subprocess.CompletedProcess:
                """Execute command in virt env."""
                activate_env_cmd = f"source {venv}/bin/activate"
                cmds = "\n".join([activate_env_cmd, cmd])
                print(f"Run command: {cmds}")
                return subprocess.run(
                    cmds, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

            for cmd in [
                "mlia --help",
                "mlia ops --help",
                "mlia perf --help",
                "aiet --help",
                "aiet system list",
                "aiet software list",
            ]:
                result = run_in_env(cmd)
                assert result.returncode == 0
                print(result.stdout.decode())

    @pytest.mark.parametrize("command", ["operators", "performance"])
    @pytest.mark.parametrize(
        "model",
        [
            "ds_cnn_large_fully_quantized_int8.tflite",
            "mobilenet_v2_1.0_224_INT8.tflite",
            "wav2letter_leakyrelu_fixed.tflite",
            "inception_v3_quant.tflite",
        ],
    )
    def test_commands_ethos_u55_real_model(self, command: str, model: str) -> None:
        """Test 'operators' and 'performance' commands on real-world TFLite models."""
        config_dir = get_config_dir()
        if not config_dir:
            raise Exception("E2E configuration directory is not provided")

        mlia_command = [
            "mlia",
            command,
            "--device",
            "ethos-u55",
            "--mac",
            "256",
            "--config",
            "tests/test_resources/vela/sample_vela.ini",
            "--system-config",
            "Ethos_U55_High_End_Embedded",
            "--memory-mode",
            "Shared_Sram",
            str(config_dir / "tflite_models" / model),
        ]

        run_command(mlia_command)
