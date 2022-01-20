# Copyright 2021, Arm Ltd.
"""End to end tests for MLIA CLI."""
# pylint: disable=no-self-use,superfluous-parens
# pylint: disable=too-many-arguments,too-many-locals,subprocess-run-check
import argparse
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import pytest
from mlia.cli.main import get_possible_command_names
from mlia.cli.main import init_commands
from mlia.cli.main import init_common_parser
from mlia.cli.main import init_subcommand_parser
from mlia.utils.general import is_list_of
from mlia.utils.proc import working_directory


pytestmark = pytest.mark.e2e

VALID_COMMANDS = get_possible_command_names()


class CommandExecution(NamedTuple):
    """Command execution."""

    parsed_args: argparse.Namespace
    parameters: List[str]

    def __str__(self) -> str:
        """Return string representation."""
        command = self._get_param("command")
        device = self._get_param("device")
        mac = self._get_param("mac")

        model_path = Path(self._get_param("model"))
        model = model_path.name

        opt_type = self._get_param("optimization_type", None)
        opt_target = self._get_param("optimization_target", None)
        opts = (
            f" optimization={opts}"
            if (opts := self._merge(opt_type, opt_target))
            else ""
        )

        return f"command {command}: device={device} mac={mac} model={model}{opts}"

    def _get_param(self, param: str, default: Optional[str] = "unknown") -> Any:
        return getattr(self.parsed_args, param, default)

    @staticmethod
    def _merge(value1: str, value2: str, sep: str = ",") -> str:
        """Split and merge values into a string."""
        if not value1 or not value2:
            return ""

        values = [
            f"{v1} {v2}"
            for v1, v2 in zip(str(value1).split(sep), str(value2).split(sep))
        ]

        return ",".join(values)


class ExecutionConfiguration(NamedTuple):
    """Execution configuration."""

    command: str
    parameters: Dict[str, List[List[str]]]

    @classmethod
    def from_dict(cls, exec_info: Dict) -> "ExecutionConfiguration":
        """Create instance from the dictionary."""
        if not (command := exec_info.get("command")):
            raise Exception("Command is not defined")

        if command not in VALID_COMMANDS:
            raise Exception(f"Unknown command {command}")

        if not (params := exec_info.get("parameters")):
            raise Exception(f"Command {command} should have parameters")

        assert isinstance(params, dict), "Parameters should be a dictionary"
        assert all(
            isinstance(param_group_name, str)
            and is_list_of(param_group_values, list)
            and all(is_list_of(param_list, str) for param_list in param_group_values)
            for param_group_name, param_group_values in params.items()
        ), "Execution configuration should be a dictionary of list of list of strings"

        return cls(command, params)

    @property
    def all_combinations(self) -> Iterable[List[str]]:
        """Generate all command combinations."""
        parameter_groups = self.parameters.values()
        parameter_combinations = itertools.product(*parameter_groups)

        return (
            [self.command, *itertools.chain.from_iterable(param_combination)]
            for param_combination in parameter_combinations
        )


def launch_and_wait(cmd: List[str], stdin: Optional[Any] = None) -> None:
    """Launch command and wait for the completion."""
    with subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # redirect command stderr to stdout
        universal_newlines=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise Exception("Unable to get process output")
        # redirect output of the process into current process stdout
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if (exit_code := process.poll()) != 0:
            raise Exception(f"Command failed with exit_code {exit_code}")


def run_command(cmd: List[str], cmd_input: Optional[str] = None) -> None:
    """Run command."""
    print(f"Run command: {' '.join(cmd)}")
    if cmd_input is None:
        launch_and_wait(cmd)
        return
    print(f"Command will receive next input: {repr(cmd_input)}")
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="mlia_", suffix="_test"
    ) as cmd_input_file:
        cmd_input_file.write(cmd_input)
        cmd_input_file.seek(0)
        launch_and_wait(cmd, cmd_input_file)


def get_config_dir() -> Optional[Path]:
    """Get configuration directory path."""
    if not (config_dir := os.environ.get("MLIA_E2E_CONFIG")):
        print("Config directory (MLIA_E2E_CONFIG) not set.")
        return None

    if not (config_dir_path := Path(config_dir)).is_dir():
        raise Exception(f"Wrong config directory {config_dir}")

    return config_dir_path


def get_config_file(config_filename: str = "e2e_tests_config.json") -> Optional[Path]:
    """Get path to the configuration file."""
    if not (config_dir := get_config_dir()):
        return None

    if not (config_file := config_dir / config_filename).is_file():
        raise Exception(
            f"Unable to find configuration file {config_filename} in {config_dir}"
        )
    return config_file


@pytest.mark.install
class TestInstallScript:
    """Install script tests."""

    def run_install_script_test(
        self,
        install_script: str,
        install_script_options: List,
        install_dir_path: Path,
        tmp_path: Path,
        commands: List,
        flag_download: str = None,
    ) -> None:
        """Run an install script use case."""
        install_dir_path.mkdir()

        shutil.copy2(f"scripts/{install_script}", tmp_path)

        with working_directory(tmp_path):
            run_command(["ls", "-R", str(tmp_path)])

            test_path = install_dir_path.resolve()
            venv = f"e2e_venv_{test_path.name}"
            extra_input = (
                f"{flag_download}\n"
                if flag_download in ["Y", "y", "N", "n", ""]
                else None
            )

            command = [
                f"./{install_script}",
                *install_script_options,
                "-e",
                venv,
                "-v",
            ]

            run_command(command, extra_input)

            assert Path(venv).is_dir()

            def run_in_venv(cmd: str) -> subprocess.CompletedProcess:
                """Execute command in virt env."""
                activate_env_cmd = f"source {venv}/bin/activate"
                print(f"Run command in venv: {cmd}")
                return subprocess.run(
                    f"{activate_env_cmd} ; {cmd}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    executable="/bin/bash",
                )

            for cmd in commands:
                result = run_in_venv(cmd)
                assert result.returncode == 0
                print(result.stdout.decode())

    def check_output(
        self, capsys: Any, expected_messages: Optional[List[str]] = None
    ) -> None:
        """Check the output and capture errors."""
        out, err = capsys.readouterr()
        print(out)
        print(err)

        if expected_messages:
            messages_not_found = [msg for msg in expected_messages if msg not in out]
            assert not messages_not_found, "Output checking is failed"

    command_list = [
        "mlia --version",
        "mlia --help",
        "mlia ops --help",
        "mlia perf --help",
        "aiet --version",
        "aiet --help",
        "aiet system list",
        "aiet application list",
    ]

    def test_install_new_script_use_default_fvp_paths_download_timeout(
        self,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        """Test MLIA mlia_install script using the default FVP paths."""
        test_dir = "dist1"
        install_dir_path = tmp_path / test_dir

        self.run_install_script_test(
            "mlia_install.sh", [], install_dir_path, tmp_path, []
        )

        self.check_output(capsys, ["Timed out so nothing will be downloaded"])

    def test_install_new_script_use_specific_fvp_path(
        self,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        """Test MLIA install_new script using a specific FVP path."""
        config_dir_path = get_config_dir()
        assert config_dir_path

        fvp_path = config_dir_path.resolve() / "systems/fvp_corstone_sse-300_ethos-u"
        assert fvp_path.is_dir()

        test_dir = "dist2"
        install_dir_path = tmp_path / test_dir

        self.run_install_script_test(
            "mlia_install.sh",
            ["-f", str(fvp_path), "-d", str(install_dir_path)],
            install_dir_path,
            tmp_path,
            self.command_list,
        )

        self.check_output(capsys, ["Using the local instance of the FVP"])

    @pytest.mark.parametrize(
        "flag_yn, pass_install_directory, test_dir, cap_str",
        [
            [
                "Y",
                False,
                "dist6",
                ["Successfully downloaded from developer.arm.com"],
            ],
            ["Y", True, "dist7", ["Successfully downloaded from developer.arm.com"]],
            ["N", False, "dist8", ["Nothing downloaded. Exiting"]],
            ["", False, "dist9", ["Nothing downloaded. Exiting"]],
        ],
    )
    def test_install_new_script_download_fvp(
        self,
        tmp_path: Path,
        flag_yn: str,
        pass_install_directory: bool,
        test_dir: str,
        cap_str: List[str],
        capsys: Any,
    ) -> None:
        """Test MLIA mlia_install script can download and install FVP."""
        install_dir_path = tmp_path / test_dir
        install_dir_option = []
        if pass_install_directory:
            install_dir_option = ["-d", str(install_dir_path)]

        self.run_install_script_test(
            "mlia_install.sh",
            install_dir_option,
            install_dir_path,
            tmp_path,
            self.command_list,
            flag_yn,
        )

        self.check_output(capsys, cap_str)

    @pytest.mark.parametrize(
        "model_name",
        ["simple_3_layers_model"],
    )
    def test_model_generation(self, tmp_path: Path, model_name: str) -> None:
        """Simple test for the gen_models.py script."""
        args = [
            "--output-dir",
            str(tmp_path),
            "--model-name",
            model_name,
            "--save-keras",
            "--tf-saved-model",
        ]

        subprocess.check_call(["python", "scripts/gen_models.py", *args])

        tflite_path = tmp_path / f"{model_name}.tflite"
        keras_path = tmp_path / f"{model_name}.h5"

        assert all(model_path.is_file() for model_path in [tflite_path, keras_path])

        saved_model_path = tmp_path / f"tf_model_{model_name}"
        assert saved_model_path.is_dir()

        model_files = list(saved_model_path.iterdir())
        assert len(model_files) > 0


def get_execution_definitions() -> Generator[CommandExecution, None, None]:
    """Collect all execution definitions from configuration file."""
    if (config_file := get_config_file()) is None:
        # if no configuration file provided then just return from this function
        # test will be skipped in this case
        return

    with open(config_file) as file:
        json_data = json.load(file)
    assert isinstance(json_data, dict), "JSON configuration expected to be a dictionary"

    executions = json_data.get("executions", [])
    assert is_list_of(executions, dict), "List of the dictionaries expected"

    exec_configs = (
        ExecutionConfiguration.from_dict(exec_info) for exec_info in executions
    )

    combinations = (
        command_combination
        for exec_config in exec_configs
        for command_combination in exec_config.all_combinations
    )

    common_parser = init_common_parser()
    subcommand_parser = init_subcommand_parser(common_parser)
    init_commands(subcommand_parser)

    for combination in combinations:
        try:
            # parse parameters to generate meaningful test description
            args = subcommand_parser.parse_args(combination)
        except SystemExit as err:
            raise Exception(
                f"Configuration contains invalid parameters: {combination}"
            ) from err

        yield CommandExecution(args, combination)


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
    application_dirs = get_directories(config_dir_path / "applications")

    install_aiet_artifacts(systems_dirs, application_dirs)


def install_aiet_artifacts(
    system_dirs: List[Path], application_dirs: List[Path]
) -> None:
    """Install AIET applications and systems."""
    run_command(["aiet", "--version"])

    for system in system_dirs:
        if not system.is_dir():
            raise Exception(f"Wrong directory {system}. Unable to install system")

        run_command(["aiet", "system", "install", "-s", str(system)])
        print(f"System {system} installed")

    run_command(["aiet", "system", "list"])

    for application in application_dirs:
        if not application.is_dir():
            raise Exception(
                f"Wrong directory {application}. Unable to install application"
            )

        run_command(["aiet", "application", "install", "-s", str(application)])
        print(f"Application {application} installed")

    run_command(["aiet", "application", "list"])


@pytest.mark.command
class TestEndToEnd:
    """End to end command tests."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up test class."""
        discover_and_install_aiet_artifacts()

    @pytest.mark.parametrize("command_execution", get_execution_definitions(), ids=str)
    def test_command(self, command_execution: CommandExecution) -> None:
        """Test MLIA command with the provided parameters."""
        mlia_command = ["mlia", *command_execution.parameters]

        run_command(mlia_command)


@pytest.mark.model_gen
class TestModelGeneration:
    """Model generation tests."""

    @pytest.mark.parametrize(
        "model_name",
        ["simple_3_layers_model"],
    )
    def test_model_generation(self, tmp_path: Path, model_name: str) -> None:
        """Simple test for the gen_models.py script."""
        args = [
            "--output-dir",
            str(tmp_path),
            "--model-name",
            model_name,
            "--save-keras",
            "--tf-saved-model",
        ]

        subprocess.check_call(["python", "scripts/gen_models.py", *args])

        tflite_path = tmp_path / f"{model_name}.tflite"
        keras_path = tmp_path / f"{model_name}.h5"

        assert all(model_path.is_file() for model_path in [tflite_path, keras_path])

        saved_model_path = tmp_path / f"tf_model_{model_name}"
        assert saved_model_path.is_dir()

        model_files = list(saved_model_path.iterdir())
        assert len(model_files) > 0
