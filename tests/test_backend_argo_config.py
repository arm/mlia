# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Argo config."""
from __future__ import annotations

from mlia.backend.argo.config import ArgoConfig


def test_argo_config() -> None:
    """Test for class ArgoConfig."""
    cfg = ArgoConfig()
    assert cfg.accelerator_config == "hydra"
