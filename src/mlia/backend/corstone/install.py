# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
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

from mlia.backend.config import System
from mlia.backend.install import BackendInstallation
from mlia.backend.install import CompoundPathChecker
from mlia.backend.install import Installation
from mlia.backend.install import PackagePathChecker
from mlia.backend.install import StaticPathChecker
from mlia.utils.download import DownloadArtifact
from mlia.utils.filesystem import working_directory


logger = logging.getLogger(__name__)


class CorstoneInstaller:
    """Helper class that wraps Corstone installation logic."""

    def __init__(self, name: str):
        """Define name of the Corstone installer."""
        self.name = name

    def __call__(self, eula_agreement: bool, dist_dir: Path) -> Path:
        """Install Corstone and return path to the models."""
        with working_directory(dist_dir):
            install_dir = self.name

            if self.name == "corstone-300":
                fvp = "./FVP_Corstone_SSE-300.sh"
            elif self.name == "corstone-310":
                fvp = "./FVP_Corstone_SSE-310.sh"
            else:
                raise RuntimeError(
                    f"Couldn't find fvp file during '{self.name}' installation"
                )

            try:
                fvp_install_cmd = [
                    fvp,
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
                # changed by the user hence disabling the security check for
                # this instance
                subprocess.check_call(fvp_install_cmd)  # nosec
            except subprocess.CalledProcessError as err:
                raise RuntimeError(
                    f"Error occurred during '{self.name}' installation"
                ) from err

            return dist_dir / install_dir


def get_corstone_300_installation() -> Installation:
    """Get Corstone-300 installation."""
    corstone_name = "corstone-300"
    if System.CURRENT == System.LINUX_AARCH64:
        url = (
            "https://developer.arm.com/-/media/Arm%20Developer%20Community/"
            "Downloads/OSS/FVP/Corstone-300/"
            "FVP_Corstone_SSE-300_11.22_35_Linux64_armv8l.tgz"
        )

        filename = "FVP_Corstone_SSE-300_11.22_35_Linux64_armv8l.tgz"
        version = "11.22_35"
        sha256_hash = "0414d3dccbf7037ad24df7002ff1b48975c213f3c1d44544d95033080d0f9ce3"
        expected_files = [
            "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
            "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65",
        ]
        backend_subfolder = "models/Linux64_armv8l_GCC-9.3"

    else:
        url = (
            "https://developer.arm.com/-/media/Arm%20Developer%20Community"
            "/Downloads/OSS/FVP/Corstone-300/"
            "FVP_Corstone_SSE-300_11.16_26.tgz"
        )
        filename = "FVP_Corstone_SSE-300_11.16_26.tgz"
        version = "11.16_26"
        sha256_hash = "e26139be756b5003a30d978c629de638aed1934d597dc24a17043d4708e934d7"
        expected_files = [
            "models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55",
            "models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U65",
        ]
        backend_subfolder = "models/Linux64_GCC-6.4"

    corstone_300 = BackendInstallation(
        name=corstone_name,
        description="Corstone-300 FVP",
        fvp_dir_name=corstone_name.replace("-", "_"),
        download_artifact=DownloadArtifact(
            name="Corstone-300 FVP",
            url=url,
            filename=filename,
            version=version,
            sha256_hash=sha256_hash,
        ),
        supported_platforms=["Linux"],
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=expected_files,
                backend_subfolder=backend_subfolder,
                settings={"profile": "default"},
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=[
                    "VHT_Corstone_SSE-300_Ethos-U55",
                    "VHT_Corstone_SSE-300_Ethos-U65",
                ],
                copy_source=False,
                settings={"profile": "AVH"},
            ),
        ),
        backend_installer=CorstoneInstaller(name=corstone_name),
    )

    return corstone_300


def get_corstone_310_installation() -> Installation:
    """Get Corstone-310 installation."""
    corstone_name = "corstone-310"
    if System.CURRENT == System.LINUX_AARCH64:
        url = (
            "https://developer.arm.com/-/media/Arm%20Developer%20Community"
            "/Downloads/OSS/FVP/Corstone-310/"
            "FVP_Corstone_SSE-310_11.24_13_Linux64_armv8l.tgz"
        )
        filename = "FVP_Corstone_SSE-310_11.24_13_Linux64_armv8l.tgz"
        version = "11.24_13"
        sha256_hash = "61be18564a7d70c8eb73736e433a65cc16ae4b59f9b028ae86d258e2c28af526"
        expected_files = [
            "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-310",
            "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-310_Ethos-U65",
        ]
        backend_subfolder = "models/Linux64_armv8l_GCC-9.3"

    else:
        url = (
            "https://developer.arm.com/-/media/Arm%20Developer%20Community"
            "/Downloads/OSS/FVP/Corstone-310/"
            "FVP_Corstone_SSE-310_11.24_13_Linux64.tgz"
        )
        filename = "FVP_Corstone_SSE-310_11.24_13_Linux64.tgz"
        version = "11.24_13"
        sha256_hash = "616ecc0e82067fe0684790cf99638b3496f9ead11051a58d766e8258e766c556"
        expected_files = [
            "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310",
            "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310_Ethos-U65",
        ]
        backend_subfolder = "models/Linux64_GCC-9.3"

    corstone_310 = BackendInstallation(
        name=corstone_name,
        description="Corstone-310 FVP",
        fvp_dir_name=corstone_name.replace("-", "_"),
        download_artifact=DownloadArtifact(
            name="Corstone-310 FVP",
            url=url,
            filename=filename,
            version=version,
            sha256_hash=sha256_hash,
        ),
        supported_platforms=["Linux"],
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=expected_files,
                backend_subfolder=backend_subfolder,
                settings={"profile": "default"},
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=[
                    "VHT_Corstone_SSE-310_Ethos-U55",
                    "VHT_Corstone_SSE-310_Ethos-U65",
                ],
                copy_source=False,
                settings={"profile": "AVH"},
            ),
        ),
        backend_installer=CorstoneInstaller(name=corstone_name),
    )

    return corstone_310


def get_corstone_installations() -> list[Installation]:
    """Get Corstone installations."""
    return [
        get_corstone_300_installation(),
        get_corstone_310_installation(),
    ]
