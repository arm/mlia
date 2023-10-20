# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Vulkan Model Converter backend module."""
import logging

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.backend.vulkan_model_converter.install import (
    get_vulkan_model_converter_installation,
)

logger = logging.getLogger(__name__)

registry.register(
    "vulkan-model-converter",
    BackendConfiguration(
        supported_advice=[],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.CUSTOM,
        installation=get_vulkan_model_converter_installation(),
    ),
    pretty_name="Vulkan Model Converter",
)
