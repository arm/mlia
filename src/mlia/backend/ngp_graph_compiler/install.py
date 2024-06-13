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
                "mlia/ngp-graph-compiler/"
                "nightly_2024-06-10_drage/3bac0e93e555cc_369032118d0ef9_b1377aa94e1cfb/"
                "graph_compiler_drage_release.tar.gz"
            ),
            sha256_hash=(
                "a465222f1faaca6e7df03b4dda27fea157575484a99ddea0bee459e1f84868b7"
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
