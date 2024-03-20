# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module utils/tflite_metrics."""
from __future__ import annotations

import os
import tempfile
from math import isclose
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import tensorflow as tf
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

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
            print(layer)
            print(weight.numpy())
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
    return TFLiteMetrics(tflite_file)


class TestTFLiteMetrics:
    """Tests for module TFLite_metrics."""

    @staticmethod
    def test_sparsity(metrics: TFLiteMetrics) -> None:
        """Test sparsity."""
        # Create new instance with a sample TensorFlow Lite file
        # Check sparsity calculation
        sparsity_per_layer = metrics.sparsity_per_layer()
        for name, sparsity in sparsity_per_layer.items():
            assert isclose(sparsity, 0.5), f"Layer '{name}' has incorrect sparsity."
        assert isclose(metrics.sparsity_overall(), 0.5)

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
        report_sparsity: bool,
        report_cluster_mode: ReportClusterMode,
        max_num_clusters: int,
        verbose: bool,
    ) -> None:
        """Test the summary function."""
        for metrics in [TFLiteMetrics(tflite_file), TFLiteMetrics(tflite_file, [])]:
            metrics.summary(
                report_sparsity=report_sparsity,
                report_cluster_mode=report_cluster_mode,
                max_num_clusters=max_num_clusters,
                verbose=verbose,
            )
