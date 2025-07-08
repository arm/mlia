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
    ngp_graph_compiler_installation = BackendInstallation(
        name="vulkan-model-converter",
        description="Vulkan Model Converter",
        fvp_dir_name="vulkan-model-converter",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-r54p0_00eac0/ml-sdk-model-converter-backend-0.80/latest/"
                "ml-sdk-model-converter-backend-0.80.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "c36f5425f6e7b874d1a877c20e1a3a36d8716058f24af0b810a886a95dbc14c1"
            ),
            header_gen_fn=artifactory_credential_headers,
        )
        + DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-r54p0_00eac0/ml-sdk-model-converter-frontend-for-tflite/latest/"
                "ml-sdk-model-converter-frontend-for-tflite.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "5a7724625dd87f7d145bceabbc4caa73ea4b2c760dd8fdb870e55aebe85e0580"
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
