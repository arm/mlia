# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Arm NN TensorFlow Lite delegate backend module."""
from typing import cast

from mlia.backend.armnn_tflite_delegate.compat import ARMNN_TFLITE_DELEGATE
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

registry.register(
    "armnn-tflite-delegate",
    BackendConfiguration(
        supported_advice=[AdviceCategory.COMPATIBILITY],
        supported_systems=None,
        backend_type=BackendType.BUILTIN,
    ),
    pretty_name=cast(str, ARMNN_TFLITE_DELEGATE["backend"]),
)
