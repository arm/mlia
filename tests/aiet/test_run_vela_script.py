# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-outer-name,no-self-use
"""Module for testing run_vela.py script."""
from pathlib import Path
from typing import Any
from typing import List

import pytest
from click.testing import CliRunner

from aiet.cli.common import MiddlewareExitCode
from aiet.resources.tools.vela.check_model import get_model_from_file
from aiet.resources.tools.vela.check_model import is_vela_optimised
from aiet.resources.tools.vela.run_vela import run_vela


@pytest.fixture(scope="session")
def vela_config_path(test_tools_path: Path) -> Path:
    """Return test systems path in a pytest fixture."""
    return test_tools_path / "vela" / "vela.ini"


@pytest.fixture(
    params=[
        ["ethos-u65-256", "Ethos_U65_High_End", "U65_Shared_Sram"],
        ["ethos-u55-32", "Ethos_U55_High_End_Embedded", "U55_Shared_Sram"],
    ]
)
def ethos_config(request: Any) -> Any:
    """Fixture to provide different configuration for Ethos-U optimization with Vela."""
    return request.param


# pylint: disable=too-many-arguments
def generate_args(
    input_: Path,
    output: Path,
    cfg: Path,
    acc_config: str,
    system_config: str,
    memory_mode: str,
) -> List[str]:
    """Generate arguments that can be passed to script 'run_vela'."""
    return [
        "-i",
        str(input_),
        "-o",
        str(output),
        "--config",
        str(cfg),
        "--accelerator-config",
        acc_config,
        "--system-config",
        system_config,
        "--memory-mode",
        memory_mode,
        "--optimise",
        "Performance",
    ]


def check_run_vela(
    cli_runner: CliRunner, args: List, expected_success: bool, output_file: Path
) -> None:
    """Run Vela with the given arguments and check the result."""
    result = cli_runner.invoke(run_vela, args)
    success = result.exit_code == MiddlewareExitCode.SUCCESS
    assert success == expected_success
    if success:
        model = get_model_from_file(output_file)
        assert is_vela_optimised(model)


def run_vela_script(
    cli_runner: CliRunner,
    input_model_file: Path,
    output_model_file: Path,
    vela_config: Path,
    expected_success: bool,
    acc_config: str,
    system_config: str,
    memory_mode: str,
) -> None:
    """Run the command 'run_vela' on the command line."""
    args = generate_args(
        input_model_file,
        output_model_file,
        vela_config,
        acc_config,
        system_config,
        memory_mode,
    )
    check_run_vela(cli_runner, args, expected_success, output_model_file)


class TestRunVelaCli:
    """Test the command-line execution of the run_vela command."""

    def test_non_optimised_model(
        self,
        cli_runner: CliRunner,
        non_optimised_input_model_file: Path,
        tmp_path: Path,
        vela_config_path: Path,
        ethos_config: List,
    ) -> None:
        """Verify Vela is run correctly on an unoptimised model."""
        run_vela_script(
            cli_runner,
            non_optimised_input_model_file,
            tmp_path / "test.tflite",
            vela_config_path,
            True,
            *ethos_config,
        )

    def test_optimised_model(
        self,
        cli_runner: CliRunner,
        optimised_input_model_file: Path,
        tmp_path: Path,
        vela_config_path: Path,
        ethos_config: List,
    ) -> None:
        """Verify Vela is run correctly on an already optimised model."""
        run_vela_script(
            cli_runner,
            optimised_input_model_file,
            tmp_path / "test.tflite",
            vela_config_path,
            True,
            *ethos_config,
        )

    def test_invalid_model(
        self,
        cli_runner: CliRunner,
        invalid_input_model_file: Path,
        tmp_path: Path,
        vela_config_path: Path,
        ethos_config: List,
    ) -> None:
        """Verify an error is raised when the input model is not valid."""
        run_vela_script(
            cli_runner,
            invalid_input_model_file,
            tmp_path / "test.tflite",
            vela_config_path,
            False,
            *ethos_config,
        )
