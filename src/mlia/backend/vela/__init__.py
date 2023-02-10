# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

registry.register(
    "vela",
    BackendConfiguration(
        supported_advice=[
            AdviceCategory.COMPATIBILITY,
            AdviceCategory.PERFORMANCE,
            AdviceCategory.OPTIMIZATION,
        ],
        supported_systems=[
            System.LINUX_AMD64,
            System.LINUX_AARCH64,
            System.WINDOWS_AMD64,
            System.WINDOWS_AARCH64,
        ],
        backend_type=BackendType.BUILTIN,
    ),
    pretty_name="Vela",
)
