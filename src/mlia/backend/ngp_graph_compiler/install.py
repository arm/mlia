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
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/mlia/ngp-2024-11-18/graph_compiler_drage_release/a26d54081c0b6a_dd79d3d93f43a5/graph_compiler_drage_release.tar.gz"
                # pylint: enable=line-too-long
            ),
            sha256_hash=(
                "445e92a787f8e806662744bc42d55f172baef08a600223bf84429458d42a4327"
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
