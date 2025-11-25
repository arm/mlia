# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module optimizations/pruning."""
from __future__ import annotations

from pathlib import Path

import pytest
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107
from numpy.core.numeric import isclose

from mlia.nn.tensorflow.optimizations.pruning import PrunableLayerPolicy
from mlia.nn.tensorflow.optimizations.pruning import Pruner
from mlia.nn.tensorflow.optimizations.pruning import PruningConfiguration
from mlia.nn.tensorflow.tflite_convert import convert_to_tflite
from mlia.nn.tensorflow.tflite_metrics import TFLiteMetrics
from tests.utils.common import get_dataset
from tests.utils.common import train_model


def _test_sparsity(
    metrics: TFLiteMetrics,
    target_sparsity: float,
    layers_to_prune: list[str] | None,
) -> None:
    pruned_sparsity_dict = metrics.sparsity_per_layer()
    num_sparse_layers = 0
    num_optimizable_layers = len(pruned_sparsity_dict)
    error_margin = 0.03
    if layers_to_prune:
        expected_num_sparse_layers = len(layers_to_prune)
    else:
        expected_num_sparse_layers = num_optimizable_layers
    for layer_name in pruned_sparsity_dict:
        if abs(pruned_sparsity_dict[layer_name] - target_sparsity) < error_margin:
            num_sparse_layers = num_sparse_layers + 1
    # make sure we are having exactly as many sparse layers as we wanted
    assert num_sparse_layers == expected_num_sparse_layers


def _test_check_sparsity(base_tflite_metrics: TFLiteMetrics) -> None:
    """Assert the sparsity of a model is zero."""
    base_sparsity_dict = base_tflite_metrics.sparsity_per_layer()
    for layer_name, sparsity in base_sparsity_dict.items():
        assert isclose(
            sparsity, 0, atol=1e-2
        ), f"Sparsity for layer '{layer_name}' is {sparsity}, but should be zero."


def _get_tflite_metrics(
    path: Path, tflite_fn: str, model: keras.Model
) -> TFLiteMetrics:
    """Save model as TFLiteModel and return metrics."""
    temp_file = path / tflite_fn
    convert_to_tflite(model, output_path=temp_file)
    return TFLiteMetrics(str(temp_file))


@pytest.mark.slow
@pytest.mark.parametrize("target_sparsity", (0.5, 0.9))
@pytest.mark.parametrize("mock_data", (False, True))
@pytest.mark.parametrize("layers_to_prune", (["conv1"], ["conv1", "conv2"], None))
def test_prune_simple_model_fully(
    target_sparsity: float,
    mock_data: bool,
    layers_to_prune: list[str] | None,
    tmp_path: Path,
    test_keras_model: Path,
) -> None:
    """Simple MNIST test to see if pruning works correctly."""
    x_train, y_train = get_dataset()
    batch_size = 1
    num_epochs = 1

    base_model = keras.models.load_model(str(test_keras_model))
    train_model(base_model)

    base_tflite_metrics = _get_tflite_metrics(
        path=tmp_path,
        tflite_fn="test_prune_simple_model_fully_before.tflite",
        model=base_model,
    )

    # Make sure sparsity is zero before pruning
    _test_check_sparsity(base_tflite_metrics)

    if mock_data:
        pruner = Pruner(
            base_model,
            PruningConfiguration(
                target_sparsity,
                layers_to_prune,
            ),
        )

    else:
        pruner = Pruner(
            base_model,
            PruningConfiguration(
                target_sparsity,
                layers_to_prune,
                x_train,
                y_train,
                batch_size,
                num_epochs,
            ),
        )

    pruner.apply_optimization()
    pruned_model = pruner.get_model()

    pruned_tflite_metrics = _get_tflite_metrics(
        path=tmp_path,
        tflite_fn="test_prune_simple_model_fully_after.tflite",
        model=pruned_model,
    )

    _test_sparsity(pruned_tflite_metrics, target_sparsity, layers_to_prune)


def test_pruneable_layer_policy_failures() -> None:
    """Test for failure conditions in the PrunableLayerPolicy class."""
    plp = PrunableLayerPolicy()

    # pylint: disable=too-few-public-methods
    class Layer:
        """Test layer class"""

        def __init__(self) -> None:
            self.name = "Layer"

    # pylint: enable=too-few-public-methods

    assert plp.allow_pruning(Layer()) is False

    with pytest.raises(
        ValueError,
        match="Models that are not part of the keras.Model "
        + "base class are not supported currently.",
    ):
        plp.ensure_model_supports_pruning(20)

    with pytest.raises(ValueError, match="Unbuilt models are not supported currently."):
        plp.ensure_model_supports_pruning(keras.Model())
