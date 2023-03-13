# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data collection module."""
from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.data_collection import HydraPerformance


def test_hydra_data_collection(
    test_tflite_model: Path,
    test_keras_model: Path,
) -> None:
    """Test Hydra data collection."""
    perf = HydraPerformance(
        model=test_tflite_model, cfg=HydraConfiguration(target="hydra")
    )
    perf.set_context(ExecutionContext(advice_category={AdviceCategory.PERFORMANCE}))
    assert perf.name() == "hydra_performance"

    with pytest.raises(NotImplementedError):
        perf.collect_data()

    perf.model = test_keras_model
    with pytest.raises(Exception):
        perf.collect_data()
