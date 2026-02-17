# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for python package based installations."""

from __future__ import annotations

from mlia.backend.install import Installation, PyPackageBackendInstallation


def get_vela_installation() -> Installation:
    """Get Vela installation."""
    return PyPackageBackendInstallation(
        name="vela",
        description="Neural network model compiler for Arm Ethos-U NPUs",
        packages_to_install=["ethos-u-vela"],
        packages_to_uninstall=["ethos-u-vela"],
        expected_packages=["ethos-u-vela"],
    )
