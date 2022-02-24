# Copyright (C) 2021-2022, Arm Ltd.
"""Tests for the module performance."""
from pathlib import Path

from mlia.core.performance import estimate_performance
from mlia.core.performance import PerformanceEstimator


def test_estimate_performance(tmp_path: Path) -> None:
    """Test function estimate_performance."""
    model_path = tmp_path / "original.tflite"

    class SampleEstimator(PerformanceEstimator[Path, int]):
        """Sample estimator."""

        def estimate(self, model: Path) -> int:
            """Estimate performance."""
            if model.name == "original.tflite":
                return 1

            return 2

    def optimized_model(_original: Path) -> Path:
        """Return path to the 'optimized' model."""
        return tmp_path / "optimized.tflite"

    results = estimate_performance(model_path, SampleEstimator(), [optimized_model])
    assert results == [1, 2]
