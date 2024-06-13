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
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/"
                "mlia/vulkan-model-converter/"
                "nightly_2024-06-10/9fb5fad54085cb_12855f1da4d0f4_df962081abf254/"
                "vulkan-ml-sdk-dev-converter.tar.gz"
            ),
            sha256_hash=(
                "a11c9f567d68c45aa98d0c01bf04e9b64bbba1903c3630202592b57b39871e03"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "back-end/vulkan-converter-back-end",
                "front-ends/tflite/vulkan-converter-tflite-front-end",
            ],
        ),
        backend_installer=None,
    )

    return ngp_graph_compiler_installation
