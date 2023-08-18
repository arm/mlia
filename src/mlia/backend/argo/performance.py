# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for Argo performance estimation."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from typing import cast

import docker
from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.config import CONFIG_TO_CLI_OPTION
from mlia.backend.argo.install import DOCKER_IMAGE_NAME
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


ARGO_OUTPUT_DIR = Path("argo-output")


def get_argo_backend_path() -> str | None:
    """Determine whether we should use Docker wrapper or Subprocess to run Argo."""
    mlia_backend_argo_path = os.environ.get("MLIA_BACKEND_ARGO_PATH")
    return mlia_backend_argo_path


def create_argo_command(
    model_path: Path, output_path: Path, backend_config: dict
) -> list:
    """Return an Argo command."""
    command = [
        str(model_path),
        "--output-dir",
        str(output_path),
    ]

    argo_cfg = ArgoConfig(**backend_config.get("argo", {}))
    for name, value in vars(argo_cfg).items():
        if not value:
            continue
        command.extend((CONFIG_TO_CLI_OPTION[name], str(value)))

    return command


def run_argo_in_docker(
    argo_args: dict,
    mlia_backend_argo_path: str | None,  # pylint: disable=unused-argument
) -> Any:
    """Run Argo using Docker wrapper."""
    mounted_model_dir = Path("/models")
    mounted_output_dir = Path("/output")

    command = ["./build-release/src/argo"] + create_argo_command(
        mounted_model_dir / argo_args["model_path"].name,
        mounted_output_dir,
        argo_args["backend_config"],
    )
    volumes = {
        str(argo_args["model_path"].resolve().parent): {
            "bind": str(mounted_model_dir),
            "mode": "ro",
        },
        argo_args["tmp_dir"]: {
            "bind": str(mounted_output_dir),
            "mode": "rw",
        },
    }
    client = docker.from_env()  # type: ignore
    logger.debug("Argo command: %s", command)
    output = client.containers.run(
        image=DOCKER_IMAGE_NAME,
        command=command,
        volumes=volumes,
        stdout=True,
        stderr=True,
        remove=True,
    )
    output = cast(bytes, output)
    return output


def run_argo_in_subprocess(argo_args: dict, mlia_backend_argo_path: str | None) -> Any:
    """Run Argo using the Subprocess module."""
    model_path = argo_args["model_path"]
    command = [mlia_backend_argo_path] + create_argo_command(
        model_path, argo_args["tmp_dir"], argo_args["backend_config"]
    )
    logger.debug("Argo command: %s", command)
    output = subprocess.check_output(command)  # nosec
    return output


def estimate_performance(
    model_path: Path, output_dir: Path, backend_config: dict
) -> Path:
    """Run Argo and return the path to the Argo stats file."""
    # Use a temporary directory for the output, because docker creates files as
    # root user, which create problems when inside the MLIA output directory.
    # The files are copied from the tmp dir into the MLIA output directory.
    # For the same reason we cannot remove the tmp dir after use.
    tmp_dir = mkdtemp(prefix="mlia-argo-")

    # pass this to the argo call
    # rename to get_argo_backend_path
    mlia_backend_argo_path = get_argo_backend_path()

    if mlia_backend_argo_path:
        run_argo = run_argo_in_subprocess
    else:
        run_argo = run_argo_in_docker

    argo_output_dir_path = output_dir / ARGO_OUTPUT_DIR

    argo_args = {
        "model_path": model_path,
        "backend_config": backend_config,
        "tmp_dir": tmp_dir,
    }

    with log_action("Running Argo performance estimation..."):
        output = run_argo(argo_args, mlia_backend_argo_path)
        logger.info(output.decode("utf-8"))

    with log_action("Verify Argo output files..."):
        logger.debug(
            "Copying Argo output files from '%s' to '%s'.",
            tmp_dir,
            argo_output_dir_path,
        )
        shutil.rmtree(str(argo_output_dir_path), ignore_errors=True)
        shutil.copytree(str(tmp_dir), str(argo_output_dir_path))
        argo_metrics_file = (
            argo_output_dir_path / f"{model_path.stem}_chrome_trace.json"
        )
        assert argo_metrics_file.is_file(), f"{argo_metrics_file} is not a file!"

    return argo_metrics_file
