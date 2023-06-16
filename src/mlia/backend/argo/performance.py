# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for Argo performance estimation."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import cast

import docker
from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.config import CONFIG_TO_CLI_OPTION
from mlia.backend.argo.install import DOCKER_IMAGE_NAME
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


ARGO_OUTPUT_DIR = Path("argo-output")


def estimate_performance(
    model_path: Path, output_dir: Path, backend_config: dict
) -> Path:
    """Run Argo and return the path to the Argo stats file."""
    mounted_model_dir = Path("/models")
    mounted_output_dir = Path("/output")
    command = [
        "./build-release/src/argo",
        f"{mounted_model_dir / model_path.name}",
        "--output-dir",
        str(mounted_output_dir),
    ]
    argo_cfg = ArgoConfig(**backend_config.get("argo", {}))
    for name, value in vars(argo_cfg).items():
        if not value:
            continue
        command.extend((CONFIG_TO_CLI_OPTION[name], str(value)))

    # Use a temporary directory for the output, because docker creates files as
    # root user, which create problems when inside the MLIA output directory.
    # The files are copied from the tmp dir into the MLIA output directory.
    # For the same reason we cannot remove the tmp dir after use.
    tmp_dir = mkdtemp(prefix="mlia-argo-")

    with log_action("Running Argo performance estimation..."):
        logger.debug("Argo command: %s", command)
        client = docker.from_env()  # type: ignore
        output = client.containers.run(
            image=DOCKER_IMAGE_NAME,
            command=command,
            volumes={
                str(model_path.resolve().parent): {
                    "bind": str(mounted_model_dir),
                    "mode": "ro",
                },
                tmp_dir: {"bind": str(mounted_output_dir), "mode": "rw"},
            },
            stdout=True,
            stderr=True,
            remove=True,
        )
        output = cast(bytes, output)
        logger.info(output.decode("utf-8"))

    argo_output_dir_path = output_dir / ARGO_OUTPUT_DIR

    # Copy the temporary directory to the MLIA output dir
    with log_action("Copy Argo output files..."):
        logger.debug(
            "Copying Argo output files from '%s' to '%s'.",
            tmp_dir,
            argo_output_dir_path,
        )
        shutil.copytree(tmp_dir, str(argo_output_dir_path))

    argo_metrics_file = argo_output_dir_path / f"{model_path.stem}_chrome_trace.json"
    assert argo_metrics_file.is_file(), f"{argo_metrics_file} is not a file!"
    return argo_metrics_file
