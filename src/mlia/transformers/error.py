# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Transformer-specific exceptions."""

from mlia.core.errors import ConfigurationError


class TransformerNotFoundError(ConfigurationError):
    """Requested transformer is not available."""
