# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for the Argo performance module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.argo.config import ArgoConfig
from mlia.backend.argo.performance import estimate_performance


def test_estimate_performance() -> None:
    """Test for function estimate_performance()."""
    with pytest.raises(NotImplementedError):
        estimate_performance(Path("DOES_NOT_EXIST"), ArgoConfig())
