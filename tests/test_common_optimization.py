# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the common optimization module."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.context import ExecutionContext
from mlia.nn.common import Optimizer
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.target.common.optimization import OptimizingDataCollector
from mlia.target.config import TargetProfile


class FakeOptimizer(Optimizer):
    """Optimizer for testing purposes."""

    def __init__(self, optimized_model_path: Path) -> None:
        """Initialize."""
        super().__init__()
        self.optimized_model_path = optimized_model_path
        self.invocation_count = 0

    def apply_optimization(self) -> None:
        """Count the invocations."""
        self.invocation_count += 1

    def get_model(self) -> TFLiteModel:
        """Return optimized model."""
        return TFLiteModel(self.optimized_model_path)

    def optimization_config(self) -> str:
        """Return something: doesn't matter, not used."""
        return ""


def test_optimizing_data_collector(
    test_keras_model: Path,
    test_tflite_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OptimizingDataCollector, base support for various targets."""
    optimizations = [
        [
            {"optimization_type": "fake", "optimization_target": 42},
        ]
    ]
    context = ExecutionContext(
        config_parameters={"common_optimizations": {"optimizations": optimizations}}
    )

    target_profile = MagicMock(spec=TargetProfile)

    fake_optimizer = FakeOptimizer(test_tflite_model)

    monkeypatch.setattr(
        "mlia.target.common.optimization.get_optimizer",
        MagicMock(return_value=fake_optimizer),
    )

    collector = OptimizingDataCollector(test_keras_model, target_profile)

    collector.set_context(context)
    collector.collect_data()

    assert fake_optimizer.invocation_count == 1
