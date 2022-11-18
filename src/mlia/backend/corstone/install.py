# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for Corstone based FVPs.

The import of subprocess module raises a B404 bandit error. MLIA usage of
subprocess is needed and can be considered safe hence disabling the security
check.
"""
from __future__ import annotations

import logging
import subprocess  # nosec
from pathlib import Path

from mlia.backend.executor.runner import BackendRunner
from mlia.backend.install import BackendInstallation
from mlia.backend.install import BackendMetadata
from mlia.backend.install import CompoundPathChecker
from mlia.backend.install import Installation
from mlia.backend.install import PackagePathChecker
from mlia.backend.install import StaticPathChecker
from mlia.utils.download import DownloadArtifact
from mlia.utils.filesystem import working_directory


logger = logging.getLogger(__name__)


class Corstone300Installer:
    """Helper class that wraps Corstone 300 installation logic."""

    def __call__(self, eula_agreement: bool, dist_dir: Path) -> Path:
        """Install Corstone-300 and return path to the models."""
        with working_directory(dist_dir):
            install_dir = "corstone-300"
            try:
                fvp_install_cmd = [
                    "./FVP_Corstone_SSE-300.sh",
                    "-q",
                    "-d",
                    install_dir,
                ]
                if not eula_agreement:
                    fvp_install_cmd += [
                        "--nointeractive",
                        "--i-agree-to-the-contained-eula",
                    ]

                # The following line raises a B603 error for bandit. In this
                # specific case, the input is pretty much static and cannot be
                # changed byt the user hence disabling the security check for
                # this instance
                subprocess.check_call(fvp_install_cmd)  # nosec
            except subprocess.CalledProcessError as err:
                raise Exception(
                    "Error occurred during Corstone-300 installation"
                ) from err

            return dist_dir / install_dir


def get_corstone_300_installation() -> Installation:
    """Get Corstone-300 installation."""
    corstone_300 = BackendInstallation(
        backend_runner=BackendRunner(),
        # pylint: disable=line-too-long
        metadata=BackendMetadata(
            name="Corstone-300",
            description="Corstone-300 FVP",
            system_config="backend_configs/systems/corstone-300/backend-config.json",
            apps_resources=[],
            fvp_dir_name="corstone_300",
            download_artifact=DownloadArtifact(
                name="Corstone-300 FVP",
                url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.16_26.tgz",
                filename="FVP_Corstone_SSE-300_11.16_26.tgz",
                version="11.16_26",
                sha256_hash="e26139be756b5003a30d978c629de638aed1934d597dc24a17043d4708e934d7",
            ),
            supported_platforms=["Linux"],
        ),
        # pylint: enable=line-too-long
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=[
                    "models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55",
                    "models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U65",
                ],
                backend_subfolder="models/Linux64_GCC-6.4",
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=[
                    "VHT_Corstone_SSE-300_Ethos-U55",
                    "VHT_Corstone_SSE-300_Ethos-U65",
                ],
                copy_source=False,
                system_config=(
                    "backend_configs/systems/corstone-300-vht/backend-config.json"
                ),
            ),
        ),
        backend_installer=Corstone300Installer(),
    )

    return corstone_300


def get_corstone_310_installation() -> Installation:
    """Get Corstone-310 installation."""
    corstone_310 = BackendInstallation(
        backend_runner=BackendRunner(),
        # pylint: disable=line-too-long
        metadata=BackendMetadata(
            name="Corstone-310",
            description="Corstone-310 FVP",
            system_config="backend_configs/systems/corstone-310/backend-config.json",
            apps_resources=[],
            fvp_dir_name="corstone_310",
            download_artifact=None,
            supported_platforms=["Linux"],
        ),
        # pylint: enable=line-too-long
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=[
                    "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310",
                    "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310_Ethos-U65",
                ],
                backend_subfolder="models/Linux64_GCC-9.3",
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=[
                    "VHT_Corstone_SSE-310",
                    "VHT_Corstone_SSE-310_Ethos-U65",
                ],
                copy_source=False,
                system_config=(
                    "backend_configs/systems/corstone-310-vht/backend-config.json"
                ),
            ),
        ),
        backend_installer=None,
    )

    return corstone_310


def get_corstone_installations() -> list[Installation]:
    """Get Corstone installations."""
    return [
        get_corstone_300_installation(),
        get_corstone_310_installation(),
    ]
