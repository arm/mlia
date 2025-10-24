# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA checker backend module."""
import logging
import warnings

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
