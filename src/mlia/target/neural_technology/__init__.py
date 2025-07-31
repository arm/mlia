# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Neural Technology target module."""
from mlia.target.neural_technology.advisor import (
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo


registry.register(
    "neural-technology",
    TargetInfo(
        supported_backends=["nx-graph-compiler", "vulkan-model-converter"],
        default_backends=["nx-graph-compiler"],
        advisor_factory_func=configure_and_get_neural_technology_advisor,
        target_profile_cls=NeuralTechnologyConfiguration,
    ),
)
