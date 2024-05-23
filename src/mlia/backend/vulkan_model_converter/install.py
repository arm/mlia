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
                "nightly_2024-05-22/vulkan-ml-sdk-dev-converter.tar.gz"
            ),
            sha256_hash=(
                "8439297a3735d7f1f50e96769be7861aa215393a62de2edd64a42ae265758703"
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
