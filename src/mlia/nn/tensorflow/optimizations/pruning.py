# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Contains class Pruner to prune a model to a specified sparsity.

In order to do this, we need to have a base model and corresponding training data.
We also have to specify a subset of layers we want to prune. For more details,
please refer to the documentation for TensorFlow Model Optimization Toolkit.
"""
from __future__ import annotations

import logging
import typing
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import (  # pylint: disable=no-name-in-module
    prune_registry,
)
from tensorflow_model_optimization.python.core.sparsity.keras import (  # pylint: disable=no-name-in-module
    pruning_wrapper,
)

from mlia.nn.tensorflow.optimizations.common import Optimizer
from mlia.nn.tensorflow.optimizations.common import OptimizerConfiguration

logger = logging.getLogger(__name__)


@dataclass
class PruningConfiguration(OptimizerConfiguration):
    """Pruning configuration."""

    optimization_target: float
    layers_to_optimize: list[str] | None = None
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    batch_size: int = 1
    num_epochs: int = 1

    def __str__(self) -> str:
        """Return string representation of the configuration."""
        return f"pruning: {self.optimization_target}"

    def has_training_data(self) -> bool:
        """Return True if training data provided."""
        return self.x_train is not None and self.y_train is not None


@dataclass
class PrunableLayerPolicy(tfmot.sparsity.keras.PruningPolicy):
    """A policy to skip unsupported layers.

    PrunableLayerPolicy makes sure that all layers subject for pruning
    are compatible with the pruning API, and that the model supports pruning.
    """

    def allow_pruning(self, layer: tf.keras.layers.Layer) -> Any:
        """Allow pruning only for layers that are prunable.

        Checks the PruneRegistry in TensorFlow Model Optimization Toolkit.
        """
        layer_is_supported = prune_registry.PruneRegistry.supports(layer)
        if not layer_is_supported:
            logger.warning(
                "Layer %s is not supported for pruning, will be skipped.", layer.name
            )

        return layer_is_supported

    def ensure_model_supports_pruning(self, model: tf.keras.Model) -> None:
        """Ensure that the model contains only supported layers."""
        # Check whether the model is a Keras model.
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "Models that are not part of the \
                            tf.keras.Model base class \
                            are not supported currently."
            )

        if not model.built:
            raise ValueError("Unbuilt models are not supported currently.")


class Pruner(Optimizer):
    """
    Pruner class. Used to prune a model to a specified sparsity.

    Sample usage:
        pruner = Pruner(
            base_model,
            optimizer_configuration)

    pruner.apply_pruning()
    pruned_model = pruner.get_model()
    """

    def __init__(
        self, model: tf.keras.Model, optimizer_configuration: PruningConfiguration
    ):
        """Init Pruner instance."""
        self.model = model
        self.optimizer_configuration = optimizer_configuration

        if not optimizer_configuration.has_training_data():
            mock_x_train, mock_y_train = self._mock_train_data(1)

            self.optimizer_configuration.x_train = mock_x_train
            self.optimizer_configuration.y_train = mock_y_train

    def optimization_config(self) -> str:
        """Return string representation of the optimization config."""
        return str(self.optimizer_configuration)

    def _mock_train_data(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.random.rand(batch_size, *self.model.input_shape[1:]),
            np.random.rand(batch_size, *self.model.output_shape[1:]),
        )

    def _setup_pruning_params(self) -> dict:
        return {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0,
                final_sparsity=self.optimizer_configuration.optimization_target,
                begin_step=0,
                end_step=self.optimizer_configuration.num_epochs,
                frequency=1,
            ),
        }

    def _apply_pruning_to_layer(
        self, layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        layers_to_optimize = self.optimizer_configuration.layers_to_optimize
        assert layers_to_optimize, "List of the layers to optimize is empty"

        if layer.name not in layers_to_optimize:
            return layer

        pruning_params = self._setup_pruning_params()
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

    def _init_for_pruning(self) -> None:
        # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_layer`
        # to the layers of the model
        if not self.optimizer_configuration.layers_to_optimize:
            pruning_params = self._setup_pruning_params()
            prunable_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.model, pruning_policy=PrunableLayerPolicy(), **pruning_params
            )
        else:
            prunable_model = tf.keras.models.clone_model(
                self.model, clone_function=self._apply_pruning_to_layer
            )

        self.model = prunable_model

    def _train_pruning(self) -> None:
        loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
        self.model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

        # Model callbacks
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        # Fitting data
        self.model.fit(
            self.optimizer_configuration.x_train,
            self.optimizer_configuration.y_train,
            batch_size=self.optimizer_configuration.batch_size,
            epochs=self.optimizer_configuration.num_epochs,
            callbacks=callbacks,
            verbose=0,
        )

    @typing.no_type_check
    def _assert_sparsity_reached(self) -> None:
        for layer in self.model.layers:
            if not isinstance(layer, pruning_wrapper.PruneLowMagnitude):
                continue

            for weight in layer.layer.get_prunable_weights():
                nonzero_weights = np.count_nonzero(tf.keras.backend.get_value(weight))
                all_weights = tf.keras.backend.get_value(weight).size

                # Types need to be ignored for this function call because
                # np.testing.assert_approx_equal does not have type annotation while the
                # current context does.
                np.testing.assert_approx_equal(
                    self.optimizer_configuration.optimization_target,
                    1 - nonzero_weights / all_weights,
                    significant=2,
                )

    def _strip_pruning(self) -> None:
        self.model = tfmot.sparsity.keras.strip_pruning(self.model)

    def apply_optimization(self) -> None:
        """Apply all steps of pruning sequentially."""
        self._init_for_pruning()
        self._train_pruning()
        self._assert_sparsity_reached()
        self._strip_pruning()

    def get_model(self) -> tf.keras.Model:
        """Get model."""
        return self.model
