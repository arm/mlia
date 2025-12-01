# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module utils/tflite_metrics."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras
from numpy import isclose

from mlia.nn.tensorflow.tflite_metrics import calculate_num_unique_weights_per_axis
from mlia.nn.tensorflow.tflite_metrics import ReportClusterMode
from mlia.nn.tensorflow.tflite_metrics import TFLiteMetrics


def _sample_keras_model() -> keras.Model:
    # Create a sample model
    keras_model = keras.Sequential(
        [
            keras.Input(shape=(8, 8, 3)),
            keras.layers.Conv2D(4, 3),
            keras.layers.DepthwiseConv2D(3),
            keras.layers.Flatten(),
            keras.layers.Dense(8),
        ]
    )
    return keras_model


def _sparse_binary_keras_model() -> keras.Model:
    def get_sparse_weights(shape: list[int]) -> np.ndarray:
        weights = np.zeros(shape)
        with np.nditer(weights, op_flags=[["writeonly"]]) as weight_it:
            for idx, value in enumerate(weight_it):
                if idx % 2 == 0:
                    value[...] = 1.0  # type: ignore
        return weights

    keras_model = _sample_keras_model()
    # Assign weights to have 0.5 sparsity
    for layer in keras_model.layers:
        if not isinstance(layer, keras.layers.Flatten):
            weight = layer.weights[0]
            weight.assign(get_sparse_weights(weight.shape))
    return keras_model


@pytest.fixture(scope="class", name="tflite_file")
def fixture_tflite_file() -> Generator:
    """Generate temporary TensorFlow Lite file for tests."""
    converter = tf.lite.TFLiteConverter.from_keras_model(_sparse_binary_keras_model())
    tflite_model = converter.convert()
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "test.tflite")
        Path(file).write_bytes(tflite_model)
        yield file


@pytest.fixture(scope="function", name="metrics")
def fixture_metrics(tflite_file: str) -> TFLiteMetrics:
    """Generate metrics file for a given TensorFlow Lite model."""
    metrics = TFLiteMetrics(tflite_file)
    metrics.filtered_details["Example"] = {
        "dtype": np.float32,
        "name": "model/Example",
        "quantization": (0.0, 0),
        "quantization_parameters": {
            "quantized_dimension": 0,
            "scales": np.array([]),
            "zero_points": np.array([0, 1]),
        },
        "shape": np.array([12]),
        "shape_signature": np.array([]),
        "sparsity_parameters": {},
        "index": 3,
    }
    # breakpoint()
    return metrics


class TestTFLiteMetrics:
    """Tests for module TFLite_metrics."""

    @staticmethod
    def test_sparsity(metrics: TFLiteMetrics) -> None:
        """Test sparsity."""
        # Create new instance with a sample TensorFlow Lite file
        # Check sparsity calculation
        sparsity_per_layer = metrics.sparsity_per_layer()
        total_sparsity = 0.0

        for name, sparsity in sparsity_per_layer.items():
            if name in [
                "sequential/conv2d/Conv2D",
                "arith.constant",
                "arith.constant1",
            ]:
                assert isclose(sparsity, 0.5), f"Layer '{name}' has incorrect sparsity."
            total_sparsity += sparsity
        assert isclose(metrics.sparsity_overall(), total_sparsity / 7, atol=0.1)

    @staticmethod
    def test_clusters(metrics: TFLiteMetrics) -> None:
        """Test clusters."""
        # NUM_CLUSTERS_PER_AXIS and NUM_CLUSTERS_MIN_MAX can be handled together
        for mode in [
            ReportClusterMode.NUM_CLUSTERS_PER_AXIS,
            ReportClusterMode.NUM_CLUSTERS_MIN_MAX,
        ]:
            num_unique_weights = metrics.num_unique_weights(mode)
            for name, num_unique_per_axis in num_unique_weights.items():
                if name in [
                    "sequential/conv2d/Conv2D",
                    "arith.constant",
                    "arith.constant1",
                ]:
                    for num_unique in num_unique_per_axis:
                        assert (
                            num_unique == 2
                        ), f"Layer '{name}' has incorrect number of clusters."
        # NUM_CLUSTERS_HISTOGRAM
        hists = metrics.num_unique_weights(ReportClusterMode.NUM_CLUSTERS_HISTOGRAM)
        assert hists
        for name, hist in hists.items():
            assert hist
            for idx, num_axes in enumerate(hist):
                if name in [
                    "sequential/conv2d/Conv2D",
                    "arith.constant",
                    "arith.constant1",
                ]:
                    # The histogram starts with the bin for for num_clusters == 1
                    num_clusters = idx + 1
                    msg = (
                        f"Histogram of layer '{name}': There are {num_axes} axes "
                        f"with {num_clusters} clusters"
                    )
                    if num_clusters == 2:
                        assert num_axes > 0, f"{msg}, but there should be at least one."
                    else:
                        assert num_axes == 0, f"{msg}, but there should be none."

    @staticmethod
    @pytest.mark.parametrize("report_sparsity", (False, True))
    @pytest.mark.parametrize("report_cluster_mode", ReportClusterMode)
    @pytest.mark.parametrize("max_num_clusters", (-1, 8))
    @pytest.mark.parametrize("verbose", (False, True))
    def test_summary(
        tflite_file: str,
        metrics: TFLiteMetrics,
        report_sparsity: bool,
        report_cluster_mode: ReportClusterMode,
        max_num_clusters: int,
        verbose: bool,
    ) -> None:
        """Test the summary function."""
        for metrics_ in [
            metrics,
            TFLiteMetrics(tflite_file),
            TFLiteMetrics(tflite_file, []),
        ]:
            metrics_.summary(
                report_sparsity=report_sparsity,
                report_cluster_mode=report_cluster_mode,
                max_num_clusters=max_num_clusters,
                verbose=verbose,
            )


def test_tflite_metrics_ignore(
    tflite_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that tensors with an empty name are being properly filtered."""
    monkeypatch.setattr(
        "tensorflow.lite.Interpreter.get_tensor_details",
        MagicMock(return_value=[{"name": ""}, {"name": "Conv2D"}]),
    )

    assert list(TFLiteMetrics(tflite_file).filtered_details.keys()) == ["Conv2D"]


@pytest.mark.parametrize(
    "weights, axis, expected_result",
    [
        (np.array([1, 2, 3, 4, 1]), 0, [1, 1, 1, 1, 1]),
        (np.array([[1, 2, 3, 4], [1, 2, 1, 2]]), 1, [1, 1, 2, 2]),
    ],
)
def test_calculate_num_unique_weights_per_axis(
    weights: np.ndarray, axis: int, expected_result: list[int]
) -> None:
    """Test for the calculate_num_unique_weights_per_axis function."""
    assert calculate_num_unique_weights_per_axis(weights, axis) == expected_result
