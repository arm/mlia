# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Arm NN TensorFlow Lite Delegate backend module."""
import logging
import warnings
from typing import cast

from mlia.backend.armnn_tflite_delegate.compat import ARMNN_TFLITE_DELEGATE
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

logger = logging.getLogger(__name__)

# Issue deprecation warning when backend module is imported
warnings.warn(
    "The ArmNN TensorFlow Lite Delegate backend is deprecated and will be removed "
    "in the next major release. This backend relies on an unmaintained project.",
    DeprecationWarning,
    stacklevel=2,
)

logger.warning(
    "ArmNN TensorFlow Lite Delegate backend is deprecated and will be removed "
    "in the next major release due to dependency on unmaintained project."
)

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
