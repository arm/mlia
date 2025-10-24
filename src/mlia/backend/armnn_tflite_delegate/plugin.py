# SPDX-FileCopyrightText: Copyright 2023,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Arm NN TensorFlow Lite delegate backend module."""
from typing import cast

from mlia.backend.armnn_tflite_delegate.compat import ARMNN_TFLITE_DELEGATE
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.registry import BackendRegistry
from mlia.core.common import AdviceCategory
from mlia.plugins.plugins import BackendPlugin


class ArmNNTFliteDelegateBackendPlugin(BackendPlugin):
    """Arm NN TensorFlow Lite Backend Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Register the backend with the registry."""
        registry.register(
            "armnn-tflite-delegate",
            BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
            ),
            pretty_name=cast(str, ARMNN_TFLITE_DELEGATE["backend"]),
        )
