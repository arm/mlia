# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for python package based installations."""
from __future__ import annotations

from mlia.backend.install import Installation
from mlia.backend.install import PyPackageBackendInstallation


def get_tosa_backend_installation() -> Installation:
    """Get TOSA backend installation."""
    return PyPackageBackendInstallation(
        name="tosa-checker",
        description="Tool to check if a ML model is compatible "
        "with the TOSA specification",
        packages_to_install=["tosa-checker"],
        packages_to_uninstall=["tosa-checker"],
        expected_packages=["tosa-checker"],
    )
