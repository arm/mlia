# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA checker backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.backend.tosa_checker.install import get_tosa_backend_installation
from mlia.core.common import AdviceCategory

registry.register(
    "tosa-checker",
    BackendConfiguration(
        supported_advice=[AdviceCategory.COMPATIBILITY],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.WHEEL,
        installation=get_tosa_backend_installation(),
    ),
    pretty_name="TOSA Checker",
)
