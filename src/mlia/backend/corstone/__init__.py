# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Corstone backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

registry.register(
    "Corstone-300",
    BackendConfiguration(
        supported_advice=[AdviceCategory.PERFORMANCE, AdviceCategory.OPTIMIZATION],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.CUSTOM,
    ),
)
registry.register(
    "Corstone-310",
    BackendConfiguration(
        supported_advice=[AdviceCategory.PERFORMANCE, AdviceCategory.OPTIMIZATION],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.CUSTOM,
    ),
)
