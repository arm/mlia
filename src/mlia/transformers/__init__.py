# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""MLIA Transformers Module."""

from mlia.transformers.registry import (
    ensure_transformer_plugins_loaded,
)

ensure_transformer_plugins_loaded()
