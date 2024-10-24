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
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-2024-10-15/vmc-backend/c4e3767e265808_68fd0277eea541_0100f6a5779831_df962081abf254_c7bebe78189804_5cf10ed0f8da71_b20ebadce18cf5_249b4871934f94/"
                "ml-sdk-model-converter-backend-0.80.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "fbff448cc50cf61dfb83c05df9582d799f3175df589668f275e8bacfa6fa5885"
            ),
            header_gen_fn=artifactory_credential_headers,
        )
        + DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-2024-10-15/vmc-frontend/68fd0277eea541_0100f6a5779831_c7bebe78189804_e843b88409039f_3f677151c843c6/"
                "ml-sdk-model-converter-frontend-for-tflite-0.80.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "138fe1a964ba3369969e81aa73344131e1587764df153208df505b8837781a1e"
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
