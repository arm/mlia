# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper to only run Vela when the input is not already optimised."""
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import click

from aiet.cli.common import ModelOptimisedException
from aiet.resources.tools.vela.check_model import check_model


def vela_output_model_path(input_model: str, output_dir: str) -> Path:
    """Construct the path to the Vela output file."""
    in_path = Path(input_model)
    tflite_vela = Path(output_dir) / f"{in_path.stem}_vela{in_path.suffix}"
    return tflite_vela


def execute_vela(vela_args: Tuple, output_dir: Path, input_model: str) -> None:
    """Execute vela as external call."""
    cmd = ["vela"] + list(vela_args)
    cmd += ["--output-dir", str(output_dir)]  # Re-add parsed out_dir to arguments
    cmd += [input_model]
    subprocess.run(cmd, check=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--input-model",
    "-i",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option("--output-model", "-o", type=click.Path(), required=True)
# Collect the remaining arguments to be directly forwarded to Vela
@click.argument("vela-args", nargs=-1, type=click.UNPROCESSED)
def run_vela(input_model: str, output_model: str, vela_args: Tuple) -> None:
    """Check input, run Vela (if needed) and copy optimised file to destination."""
    output_dir = Path(output_model).parent
    try:
        check_model(input_model)  # raises an exception if already Vela-optimised
        execute_vela(vela_args, output_dir, input_model)
        print("Vela optimisation complete.")
        src_model = vela_output_model_path(input_model, str(output_dir))
    except ModelOptimisedException as ex:
        # Input already optimized: copy input file to destination path and return
        print(f"Input already vela-optimised.\n{ex}")
        src_model = Path(input_model)
    except subprocess.CalledProcessError as ex:
        print(ex)
        raise SystemExit(ex.returncode) from ex

    try:
        shutil.copyfile(src_model, output_model)
    except (shutil.SameFileError, OSError) as ex:
        print(ex)
        raise SystemExit(ex.errno) from ex


def main() -> None:
    """Entry point of check_model application."""
    run_vela()  # pylint: disable=no-value-for-parameter
