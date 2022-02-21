# Copyright (C) 2021-2022, Arm Ltd.
"""Checks runner."""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import NoReturn
from typing import Optional


_commands = []


def filter_keys(dict_to_filter: Dict, *keys: str) -> Dict:
    """Filter dictionary."""
    return {k: v for k, v in dict_to_filter.items() if k not in keys}


def error(msg: Optional[str] = None) -> NoReturn:
    """Print error message and exit."""
    if msg:
        echo(msg)

    sys.exit(1)


def command(params: Dict) -> Callable:
    """Mark function as a command."""

    def cmd_register(func: Callable) -> Callable:
        """Register the command."""
        _commands.append((func.__name__, func.__doc__, func, params))
        return func

    return cmd_register


def echo(msg: str) -> None:
    """Print message and flush output."""
    print(msg, flush=True)


def show_label(
    label: str, prefix: str = ">>>", suffix: str = "<<<", delim: str = "-"
) -> None:
    """Show label."""
    caption_len = len(label) + len(prefix) + len(suffix) + 2
    border = delim * caption_len

    echo(border)
    echo(f"{prefix} {label} {suffix}")
    echo(border)


def shell(script: str, label: Optional[str] = None, in_venv: bool = False) -> None:
    """Run shell command."""
    enable_virtual_env = ""
    if in_venv:
        enable_virtual_env = 'source "/home/foo/v/bin/activate"'

    script_to_execute = f"""
set -e
set -u
set -o pipefail

{enable_virtual_env}

{script}
"""
    try:
        if label:
            show_label(label)

        ret_code = subprocess.call(
            script_to_execute, shell=True, executable="/bin/bash"
        )

        if ret_code != 0:
            sys.exit(ret_code)
    except KeyboardInterrupt:
        error("Execution interrupted")


def run_in_venv(script: str, label: Optional[str] = None) -> None:
    """Run script in virtual environment."""
    shell(script, label, in_venv=True)


@contextmanager
def working_directory(working_dir: Path) -> Generator[Path, None, None]:
    """Temporary change working directory."""
    current_working_dir = Path.cwd()
    os.chdir(working_dir)

    try:
        yield working_dir
    finally:
        os.chdir(current_working_dir)


def valid_directory(param: str) -> Path:
    """Check if passed string is a valid directory path."""
    if not (dir_path := Path(param)).is_dir():
        error(f"Invalid directory path {param}")

    return dir_path


def init_args_parser() -> argparse.ArgumentParser:
    """Init argument parser."""
    parser = argparse.ArgumentParser(description="MLIA tools runner")

    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    for name, help_info, func, params in _commands:
        command_parser = subparsers.add_parser(name, help=help_info)
        command_parser.set_defaults(func=func)

        for param_name in params.keys():
            command_parser.add_argument(param_name, **params[param_name])

    return parser


@command(
    params={
        "workspace": {
            "type": valid_directory,
            "help": "Path to the project workspace",
        }
    },
)
def run_checks(workspace: Path) -> None:
    """Run checks."""
    with working_directory(workspace):
        run_in_venv(
            """
pip install .
pre-commit run --all-files --hook-stage=push

python3 setup.py -q check
python3 setup.py -q sdist bdist_wheel
    """
        )


def install_aiet_wheel(workspace: Path, aiet_artifact_path: str) -> None:
    """Install AIET wheel."""
    full_aiet_path = workspace / aiet_artifact_path

    if not full_aiet_path.is_file():
        error(f"Cannot find the wheel {full_aiet_path}")

    with working_directory(workspace):
        run_in_venv(
            f'pip install "{aiet_artifact_path}"',
            label=f"Installing AIET from {aiet_artifact_path} ...",
        )


def build_and_install_mlia(workspace: Path) -> None:
    """Build and install MLIA wheel."""
    with working_directory(workspace):
        # fixed version is used for locating output wheel file
        # in dist directory, without it wheel will have some dev
        # version that hard to resolve
        fixed_wheel_version = "e2e_build"

        run_in_venv(
            f'SETUPTOOLS_SCM_PRETEND_VERSION="{fixed_wheel_version}" '
            "python -m build --wheel",
            label="Building MLIA wheel ...",
        )

        wheel_path = f"dist/mlia-{fixed_wheel_version}-py3-none-any.whl"
        run_in_venv(
            f'pip install "{wheel_path}"',
            label=f"Installing MLIA from {wheel_path} ...",
        )


def launch_e2e_tests(
    workspace: Path, include_tests: List[str], exclude_tests: List[str]
) -> None:
    """Launch e2e tests."""
    if not include_tests:
        error("No test configuration provided. Please specify what tests to run.")

    pytest_marker = "e2e"
    for item in include_tests:
        if item != "all":
            pytest_marker += f" and {item}"

    for item in exclude_tests:
        pytest_marker += f" and not {item}"

    with working_directory(workspace):
        run_in_venv(
            f"""
export PYTHONUNBUFFERED=1
pytest --collect-only -m "{pytest_marker}"
pytest -v --capture=tee-sys --durations=0 --durations-min=5 --tb=long \
       --junit-xml=report/report.xml -m "{pytest_marker}"
""",
            label="Running E2E tests ...",
        )


def extract_artifacts(workspace: Path, e2e_config: str) -> None:
    """Extract AIET artifacts."""
    show_label("Extracting the AIET artifacts ...")

    def is_tar_archive(artifact: Path) -> bool:
        """Check if file is a tar archive."""
        return artifact.is_file() and artifact.name.endswith("tar.gz")

    with working_directory(workspace):
        for dir_name in ["systems", "applications"]:
            dir_path = workspace / e2e_config / dir_name

            if not dir_path.is_dir():
                continue

            for archive in (
                item for item in dir_path.iterdir() if is_tar_archive(item)
            ):
                with tarfile.open(archive, mode="r:gz") as tar_archive:
                    print(f"Extract {archive} into {dir_path}")

                    tar_archive.extractall(dir_path)

            if dir_name == "systems":
                corstone_dir = "fvp_corstone_sse-300_ethos-u"
                if (corstone_path := dir_path / corstone_dir).is_dir():
                    shutil.copy2(
                        "src/mlia/resources/aiet/systems/cs-300/aiet-config.json",
                        corstone_path,
                    )


def parse_test_markers(markers: str, sep: str = ",") -> List[str]:
    """Parse and validate test markers."""
    if not markers:
        return []

    parsed_markers = [marker.strip() for marker in markers.split(sep)]
    valid_markers = {"all", "install", "command", "model_gen"}

    invalid_markers = [
        marker for marker in parsed_markers if marker not in valid_markers
    ]
    if invalid_markers:
        error(
            f"Invalid test markers {','.join(invalid_markers)}. "
            f"Please choose from {', '.join(valid_markers)}"
        )

    return parsed_markers


@command(
    params={
        "workspace": {
            "type": valid_directory,
            "help": "Path to the project workspace",
        },
        "--tests-to-run": {
            "default": "all",
            "help": "E2E test groups to run "
            "(comma separated list, e.g. command,install)",
        },
        "--tests-to-skip": {
            "help": "E2E test groups to skip "
            "(comma separated list, e.g. command,install)"
        },
    },
)
def run_e2e_tests(workspace: Path, tests_to_run: str, tests_to_skip: str) -> None:
    """Run e2e tests."""
    e2e_config = os.getenv("MLIA_E2E_CONFIG")
    aiet_artifact_path = os.getenv("AIET_ARTIFACT_PATH")

    show_label("Launching E2E tests ...")
    print(
        "Parameters:\n  "
        f"{e2e_config=}\n  "
        f"{aiet_artifact_path=}\n  "
        f"{tests_to_run=}\n  "
        f"{tests_to_skip=}"
    )

    include_tests = parse_test_markers(tests_to_run)
    exclude_tests = parse_test_markers(tests_to_skip)
    for item in exclude_tests:
        if item == "all":
            error("'all' cannot be in excluded tests")

    if not e2e_config:
        error(
            "Path to the configuration directory is not set. "
            "Please, set env variable MLIA_E2E_CONFIG"
        )

    e2e_config_path = workspace / e2e_config

    if not e2e_config_path.is_dir():
        error(f"{e2e_config_path} is not a directory")

    if aiet_artifact_path:
        install_aiet_wheel(workspace, aiet_artifact_path)

    build_and_install_mlia(workspace)

    extract_artifacts(workspace, e2e_config)

    launch_e2e_tests(workspace, include_tests, exclude_tests)


@command(
    params={
        "workspace": {
            "type": valid_directory,
            "help": "Path to the project workspace",
        }
    },
)
def gen_docs(workspace: Path) -> None:
    """Generate documentation."""
    with working_directory(workspace):
        run_in_venv(
            """
pip install .

WORKSPACE=$PWD
cd docs
sphinx-apidoc -f -o source "$WORKSPACE/src/mlia"
make html
    """
        )


def main() -> None:
    """Run app."""
    parser = init_args_parser()
    args = parser.parse_args()

    func_args = filter_keys(vars(args), "command", "func")
    args.func(**func_args)


if __name__ == "__main__":
    main()
