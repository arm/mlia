# Copyright 2021, Arm Ltd.
"""Test for module utils/tflite_metrics."""
# pylint: disable=no-self-use,too-many-arguments,redefined-outer-name
import os
import tempfile
from math import isclose
from typing import Generator
from typing import List

import numpy as np
import pytest
import tensorflow as tf
from mlia.utils.tflite_metrics import ReportClusterMode
from mlia.utils.tflite_metrics import TFLiteMetrics


def _dummy_keras_model() -> tf.keras.Model:
    # Create a dummy model
    keras_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(8, 8, 3)),
            tf.keras.layers.Conv2D(4, 3),
            tf.keras.layers.DepthwiseConv2D(3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8),
        ]
    )
    return keras_model


def _sparse_binary_keras_model() -> tf.keras.Model:
    def get_sparse_weights(shape: List[int]) -> np.array:
        weights = np.zeros(shape)
        with np.nditer(weights, op_flags=["writeonly"]) as weight_iterator:
            for idx, value in enumerate(weight_iterator):
                if idx % 2 == 0:
                    value[...] = 1.0
        return weights

    keras_model = _dummy_keras_model()
    # Assign weights to have 0.5 sparsity
    for layer in keras_model.layers:
        if not isinstance(layer, tf.keras.layers.Flatten):
            weight = layer.weights[0]
            weight.assign(get_sparse_weights(weight.shape))
            print(layer)
            print(weight.numpy())
    return keras_model


@pytest.fixture(scope="class")
def tflite_file() -> Generator:
    """Generate temporary tflite file for tests."""
    converter = tf.lite.TFLiteConverter.from_keras_model(_sparse_binary_keras_model())
    tflite_model = converter.convert()
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "test.tflite")
        open(file, "wb").write(tflite_model)
        yield file


@pytest.fixture(scope="function")
def metrics(tflite_file: str) -> TFLiteMetrics:
    """Generate metrics file for a given tflite model."""
    return TFLiteMetrics(tflite_file)


class TestTFLiteMetrics:
    """Tests for module tflite_metrics."""

    def test_sparsity(self, metrics: TFLiteMetrics) -> None:
        """Test sparsity."""
        # Create new instance with a dummy tflite file
        # Check sparsity calculation
        sparsity_per_layer = metrics.sparsity_per_layer()
        for name, sparsity in sparsity_per_layer.items():
            assert isclose(sparsity, 0.5), "Layer '{}' has incorrect sparsity.".format(
                name
            )
        assert isclose(metrics.sparsity_overall(), 0.5)

    def test_clusters(self, metrics: TFLiteMetrics) -> None:
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
                    ), "Layer '{}' has incorrect number of clusters.".format(name)
        # NUM_CLUSTERS_HISTOGRAM
        hists = metrics.num_unique_weights(ReportClusterMode.NUM_CLUSTERS_HISTOGRAM)
        assert hists
        for name, hist in hists.items():
            assert hist
            for idx, num_axes in enumerate(hist):
                # The histogram starts with the bin for for num_clusters == 1
                num_clusters = idx + 1
                msg = (
                    "Histogram of layer '{}': There are {} axes with {} "
                    "clusters".format(name, num_axes, num_clusters)
                )
                if num_clusters == 2:
                    assert num_axes > 0, "{}, but there should be at least one.".format(
                        msg
                    )
                else:
                    assert num_axes == 0, "{}, but there should be none.".format(msg)

    @pytest.mark.parametrize("report_sparsity", (False, True))
    @pytest.mark.parametrize("report_cluster_mode", ReportClusterMode)
    @pytest.mark.parametrize("max_num_clusters", (-1, 8))
    @pytest.mark.parametrize("verbose", (False, True))
    def test_summary(
        self,
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
