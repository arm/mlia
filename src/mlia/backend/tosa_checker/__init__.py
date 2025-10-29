# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA checker backend module."""
import logging
import warnings

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.backend.tosa_checker.install import get_tosa_backend_installation
from mlia.core.common import AdviceCategory

logger = logging.getLogger(__name__)

# Issue deprecation warning when backend module is imported
warnings.warn(
    "The TOSA Checker backend is deprecated. "
    "This backend relies on an unmaintained project.",
    DeprecationWarning,
    stacklevel=2,
)

logger.warning(
    "TOSA Checker backend is deprecated due to dependency on an unmaintained "
    "project."
)


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
