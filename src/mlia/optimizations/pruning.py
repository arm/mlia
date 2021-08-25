# Copyright 2021, Arm Ltd.
"""
Contains class Pruner to prune a model to a specified sparsity.

In order to do this, we need to have a base model and corresponding training data.
We also have to specify a subset of layers we want to prune.
"""
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from mlia.optimizations.common import Optimizer
from mlia.optimizations.common import OptimizerConfiguration
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


class PruningConfiguration(OptimizerConfiguration):
    """Pruning configuration."""

    def __init__(
        self,
        optimization_target: float,
        layers_to_optimize: Optional[List[str]] = None,
        x_train: Optional[np.array] = None,
        y_train: Optional[np.array] = None,
        batch_size: int = 1,
        num_epochs: int = 1,
    ):
        """Init pruning configuration."""
        self.optimization_target = optimization_target
        self.layers_to_optimize = layers_to_optimize
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_epochs = num_epochs


class Pruner(Optimizer):
    """
    Pruner class. Used to prune a model to a specified sparsity.

    Sample usage:
    pruner = Pruner(
        base_model,
        optimizer_configuration,
    )
    pruner.apply_pruning()
    pruned_model = pruner.get_model()
    """

    def __init__(
        self, model: tf.keras.Model, optimizer_configuration: PruningConfiguration
    ):
        """Init Pruner instance."""
        self.model = model
        self.optimizer_configuration = optimizer_configuration

        if (
            self.optimizer_configuration.x_train is None
            or self.optimizer_configuration.y_train is None
        ):
            (
                self.optimizer_configuration.x_train,
                self.optimizer_configuration.y_train,
            ) = self._mock_train_data(1)

    def _mock_train_data(self, num_imgs: int) -> Tuple[np.array, np.array]:
        input_shape = self.model.input_shape
        # get rid of the batch_size dimension
        input_shape = tuple([x for x in input_shape if x is not None])
        output_shape = self.model.output_shape
        # get rid of the batch_size dimension
        output_shape = tuple([x for x in output_shape if x is not None])
        return (
            np.random.rand(num_imgs, *input_shape),
            np.random.randint(0, output_shape[-1], (num_imgs)),
        )

    def _setup_pruning_params(self) -> dict:

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0,
                final_sparsity=self.optimizer_configuration.optimization_target,
                begin_step=0,
                end_step=self.optimizer_configuration.num_epochs,
                frequency=1,
            ),
        }

        return pruning_params

    def _apply_pruning_to_layer(
        self, layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        pruning_params = self._setup_pruning_params()

        # To make mypy happy.
        assert self.optimizer_configuration.layers_to_optimize is not None

        if layer.name in self.optimizer_configuration.layers_to_optimize:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

        return layer

    def _init_for_pruning(self) -> None:
        # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_layer`
        # to the layers of the model.

        if self.optimizer_configuration.layers_to_optimize is None:
            pruning_params = self._setup_pruning_params()
            prunable_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.model, **pruning_params
            )
        else:
            prunable_model = tf.keras.models.clone_model(
                self.model, clone_function=self._apply_pruning_to_layer
            )

        self.model = prunable_model

    def _train_pruning(self) -> None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
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

    def _assert_sparsity_reached(self) -> None:
        for layer in self.model.layers:
            if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
                for weight in layer.layer.get_prunable_weights():
                    nonzero_weights = np.count_nonzero(
                        tf.keras.backend.get_value(weight)
                    )
                    all_weights = tf.keras.backend.get_value(weight).size
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
