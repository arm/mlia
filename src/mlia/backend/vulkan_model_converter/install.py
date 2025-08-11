# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the installation of Vulkan Model Converter."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import BackendInstallation
from mlia.backend.install import PackagePathChecker
from mlia.utils.download import DownloadConfig


def get_vulkan_model_converter_installation() -> BackendInstallation:
    """Get all information to install Vulkan Model Converter."""
    nx_graph_compiler_installation = BackendInstallation(
        name="vulkan-model-converter",
        description="Vulkan Model Converter",
        fvp_dir_name="vulkan-model-converter",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/vulkan-model-converter/latest/ml-sdk-model-converter-backend-1.00.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "8c6802107a478fc2ccd4a208776401663a167e472aa3c738a90bcf4121b08d34"
            ),
            header_gen_fn=artifactory_credential_headers,
        )
        + DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/vulkan-model-converter/latest/ml-sdk-model-converter-frontend-for-tflite-1.00.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "e6364584fec17714da1745a0567d5e115abe5f33d97ae070048d9471585017ad"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "model-converter",
                "converter-tflite-frontend",
            ],
        ),
        backend_installer=None,
    )

    return nx_graph_compiler_installation
