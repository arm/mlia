# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Arm NN TensorFlow Lite delegate backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

registry.register(
    "ArmNNTFLiteDelegate",
    BackendConfiguration(
        supported_advice=[AdviceCategory.OPERATORS],
        supported_systems=None,
        backend_type=BackendType.BUILTIN,
    ),
)
