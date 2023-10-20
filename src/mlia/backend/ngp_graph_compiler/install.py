# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
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
                "https://artifactory.eu02.arm.com:443/artifactory/ml-tooling.misc/"
                "mlia/ngp-graph-compiler/regorc-20231017-9ee66a57-linux-x86_64.tar.gz"
            ),
            sha256_hash=(
                "6049f55e62887847d6a83a784d7befc834c762e74c15c177b2a1ef86bb6a82d4"
            ),
            header_gen_fn=artifactory_credential_headers,
        ),
        supported_platforms=["Linux"],
        path_checker=PackagePathChecker(
            expected_files=[
                "regorc-20231017-9ee66a57-linux-x86_64/bin/regorc-0.1.0",
                "regorc-20231017-9ee66a57-linux-x86_64/lib64/libregor.so.0.1.0",
            ],
            backend_subfolder="regorc-20231017-9ee66a57-linux-x86_64",
        ),
        backend_installer=None,
        dependencies=["vulkan-model-converter"],
    )

    return ngp_graph_compiler_installation
