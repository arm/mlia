# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra data collection module."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.hydra.data_collection import HydraPerformance


def test_hydra_data_collection(
    monkeypatch: pytest.MonkeyPatch,
    test_resources_path: Path,
    test_tflite_model: Path,
    test_keras_model: Path,
) -> None:
    """Test Hydra data collection."""
    test_metrics_file = Path(
        test_resources_path
        / "chrometrace/ds_cnn_large_fully_quantized_int8_chrome_trace.json"
    )
    monkeypatch.setattr(
        "mlia.target.hydra.performance.estimate_performance",
        MagicMock(return_value=test_metrics_file),
    )

    perf = HydraPerformance(
        model=test_tflite_model, cfg=HydraConfiguration(target="hydra")
    )
    perf.set_context(ExecutionContext(advice_category={AdviceCategory.PERFORMANCE}))
    assert perf.name() == "hydra_performance"

    metrics = perf.collect_data()
    assert metrics.metrics_file == test_metrics_file
    assert metrics.target_config.target == "hydra"

    perf.model = test_keras_model
    with pytest.raises(Exception):
        perf.collect_data()
