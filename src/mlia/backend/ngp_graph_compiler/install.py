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
                "https://artifactory.arm.com:443/artifactory/ml-tooling.misc/"
                "mlia/ngp-graph-compiler/regorc-231220-11393e21-linux-x86_64.tar.gz"
            ),
            sha256_hash=(
                "7b052627a40c90b79460b54c9e271977b705b9b2321da25c495fde6ea9bf2951"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "regorc-231220-11393e21-linux-x86_64/bin/regorc-0.1.0",
                "regorc-231220-11393e21-linux-x86_64/lib64/libregor.so.0.1.0",
            ],
            backend_subfolder="regorc-231220-11393e21-linux-x86_64",
        ),
        backend_installer=None,
        dependencies=["vulkan-model-converter"],
    )

    return ngp_graph_compiler_installation
