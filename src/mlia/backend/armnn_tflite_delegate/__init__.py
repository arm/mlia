# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Arm NN TensorFlow Lite Delegate backend module."""
import logging
import warnings

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
