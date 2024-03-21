# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for Corstone based FVPs.

The import of subprocess module raises a B404 bandit error. MLIA usage of
subprocess is needed and can be considered safe hence disabling the security
check.
"""
from __future__ import annotations

import logging
import re
import subprocess  # nosec
from pathlib import Path

from mlia.backend.config import System
from mlia.backend.install import BackendInstallation
from mlia.backend.install import CompoundPathChecker
from mlia.backend.install import Installation
from mlia.backend.install import PackagePathChecker
from mlia.backend.install import StaticPathChecker
from mlia.utils.download import DownloadConfig
from mlia.utils.filesystem import working_directory


logger = logging.getLogger(__name__)

ARM_ECOSYSTEM_FVP_URL = (
    "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/"
)


class CorstoneFVP:
    """Corstone FVP."""

    def __init__(
        self, archive: str, sha256_hash: str, fvp_expected_files: list
    ) -> None:
        """Initialize Corstone FVP."""
        self.archive: str = archive
        self.sha256_hash: str = sha256_hash
        self.fvp_expected_files: list = fvp_expected_files

    def get_fvp_version(self) -> str:
        """Get fvp version from archive name."""
        version_pattern = r"\d+\.\d+_\d+"
        match_result = re.search(version_pattern, self.archive)
        if match_result:
            return match_result.group()

        raise RuntimeError("Couldn't find Corstone FVP version.")

    def get_vht_expected_files(self) -> list:
        """Get expected files in vht."""
        return ["VHT" + fvp.split("/")[-1][3:] for fvp in self.fvp_expected_files]


# pylint: disable=line-too-long
CORSTONE_FVPS: dict[str, dict[str, CorstoneFVP]] = {
    "corstone-300": {
        "x86": CorstoneFVP(
            archive="Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz",
            sha256_hash="6ea4096ecf8a8c06d6e76e21cae494f0c7139374cb33f6bc3964d189b84539a9",
            fvp_expected_files=[
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65",
            ],
        ),
        "aarch64": CorstoneFVP(
            archive="Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64_armv8l.tgz",
            sha256_hash="9b43da6a688220c707cd1801baf9cf4f5fb37d6dc77587b9071347411a64fd56",
            fvp_expected_files=[
                "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
                "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U65",
            ],
        ),
    },
    "corstone-310": {
        "x86": CorstoneFVP(
            archive="Corstone-310/FVP_Corstone_SSE-310_11.24_13_Linux64.tgz",
            sha256_hash="616ecc0e82067fe0684790cf99638b3496f9ead11051a58d766e8258e766c556",
            fvp_expected_files=[
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310",
                "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310_Ethos-U65",
            ],
        ),
        "aarch64": CorstoneFVP(
            archive="Corstone-310/FVP_Corstone_SSE-310_11.24_13_Linux64_armv8l.tgz",
            sha256_hash="61be18564a7d70c8eb73736e433a65cc16ae4b59f9b028ae86d258e2c28af526",
            fvp_expected_files=[
                "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-310",
                "models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-310_Ethos-U65",
            ],
        ),
    },
}


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


def get_corstone_installation(corstone_name: str) -> Installation:
    """Get Corstone installation."""
    if System.CURRENT == System.LINUX_AARCH64:
        arch = "aarch64"

    elif System.CURRENT == System.LINUX_AMD64:
        arch = "x86"

    else:
        raise RuntimeError(f"'{corstone_name}' is not compatible with this platform")

    corstone_fvp = CORSTONE_FVPS[corstone_name][arch]
    archive = corstone_fvp.archive
    sha256_hash = corstone_fvp.sha256_hash
    url = ARM_ECOSYSTEM_FVP_URL + archive
    expected_files_fvp = corstone_fvp.fvp_expected_files
    expected_files_vht = corstone_fvp.get_vht_expected_files()
    backend_subfolder = expected_files_fvp[0].split("FVP")[0]

    corstone_install = BackendInstallation(
        name=corstone_name,
        description=corstone_name.capitalize() + " FVP",
        fvp_dir_name=corstone_name.replace("-", "_"),
        download_config=DownloadConfig(
            url=url,
            sha256_hash=sha256_hash,
        ),
        supported_platforms=["Linux"],
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=expected_files_fvp,
                backend_subfolder=backend_subfolder,
                settings={"profile": "default"},
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=expected_files_vht,
                copy_source=False,
                settings={"profile": "AVH"},
            ),
        ),
        backend_installer=CorstoneInstaller(name=corstone_name),
    )

    return corstone_install
