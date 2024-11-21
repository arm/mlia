# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of Vulkan Model Converter."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import BackendInstallation
from mlia.backend.install import PackagePathChecker
from mlia.utils.download import DownloadConfig


def get_vulkan_model_converter_installation() -> BackendInstallation:
    """Get all information to install Vulkan Model Converter."""
    ngp_graph_compiler_installation = BackendInstallation(
        name="vulkan-model-converter",
        description="Vulkan Model Converter",
        fvp_dir_name="vulkan-model-converter",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-2024-11-18/"
                "ml-sdk-model-converter-backend_r53-0.80/"
                "c4e3767e265808_68fd0277eea541_0100f6a5779831_df962081abf254_c7bebe78189804_434d3d9b47fe88_23e9b0f022b6e4_dee82ef0eb7b17/"
                "ml-sdk-model-converter-backend_r53-0.80.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "88887afd26b78aa227f142543b6edb04e86f4e75aa57e58e3977259266308ae2"
            ),
            header_gen_fn=artifactory_credential_headers,
        )
        + DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-2024-11-18/"
                "ml-sdk-model-converter-frontend-for-tflite-r53-0.80/"
                "68fd0277eea541_0100f6a5779831_c7bebe78189804_e843b88409039f_fd8b31a1ac6045/"
                "ml-sdk-model-converter-frontend-for-tflite-r53-0.80.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "3c59e7815fc2868a40874e1d1a102510186fde7fb254aa62cb8eb04e6c914e63"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "converter-backend",
                "converter-tflite-frontend",
            ],
        ),
        backend_installer=None,
    )

    return ngp_graph_compiler_installation
