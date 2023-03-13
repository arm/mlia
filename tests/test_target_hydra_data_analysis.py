# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data analysis module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.data_analysis import HydraDataAnalyzer
from mlia.target.hydra.performance import ArgoStats


def test_hydra_data_analyzer() -> None:
    """Test Hydra data analyzer."""
    analyzer = HydraDataAnalyzer()
    argo_stats = ArgoStats(
        device=HydraConfiguration(target="Hydra"),
        metrics_file=Path("DOES_NOT_EXIST"),
    )
    with pytest.raises(NotImplementedError):
        analyzer.analyze_data(argo_stats)
