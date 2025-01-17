# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of NGP Graph Compiler."""
from __future__ import annotations

from mlia.backend.install import artifactory_credential_headers
from mlia.backend.install import BackendInstallation
from mlia.backend.install import PackagePathChecker
from mlia.utils.download import DownloadConfig


def get_ngp_graph_compiler_installation() -> BackendInstallation:
    """Get all information to install NGP Graph Compiler."""
    ngp_graph_compiler_installation = BackendInstallation(
        name="ngp-graph-compiler",
        description="NGP Graph Compiler",
        fvp_dir_name="ngp-graph-compiler",
        download_config=DownloadConfig(
            url=(
                # pylint: disable=line-too-long
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-r54p0_00eac0/graph_compiler_drage_release/latest/graph_compiler_drage_release.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "cff890214750d8db5c80132a699c34c429e696c02a9cf1fd7a69845ff30a162e"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "regorc",
                "regorc-0.1.0",
            ],
            backend_subfolder="graph-compiler",
        ),
        backend_installer=None,
        dependencies=["vulkan-model-converter"],
    )

    return ngp_graph_compiler_installation
