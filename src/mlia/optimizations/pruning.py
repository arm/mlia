"""
Contains class Pruner to prune a model to a specified sparsity.

In order to do this, we need to have a base model and corresponding training data.
We also have to specify a subset of layers we want to prune.
"""
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


class Pruner:
    """
    Pruner class. Used to prune a model to a specified sparsity.

    Sample usage:
    pruner = Pruner(
        base_model,
        x_train,
        y_train,
        target_sparsity,
        batch_size,
        num_epochs,
        layers_to_prune,
    )
    pruner.apply_pruning()
    pruned_model = pruner.get_model()
    """

    def __init__(
        self,
        model: tf.keras.Model,
        x_train: np.array,
        y_train: np.array,
        target_sparsity: float,
        batch_size: int,
        num_epochs: int,
        layers_to_prune: List[str],
    ):
        """Init Pruner instance."""
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.target_sparsity = target_sparsity
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.layers_to_prune = layers_to_prune

    def get_model(self) -> tf.keras.Model:
        """Return the model instance from the pruner."""
        return self.model

    def _setup_pruning_params(self) -> dict:
        num_images = self.x_train.shape[0]

        end_step = (
            np.ceil(num_images / self.batch_size).astype(np.int32) * self.num_epochs
        )

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.target_sparsity, begin_step=0, end_step=end_step
            ),
        }

        return pruning_params

    def _apply_pruning_to_layer(
        self, layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        pruning_params = self._setup_pruning_params()

        if layer.name in self.layers_to_prune:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

        return layer

    def _init_for_pruning(self) -> None:
        # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_layer`
        # to the layers of the model.
        prunable_model = tf.keras.models.clone_model(
            self.model, clone_function=self._apply_pruning_to_layer
        )

        self.model = prunable_model

    def _train_pruning(self) -> None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

        # Model callbacks
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        # Fitting data
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=callbacks,
        )

    def _strip_pruning(self) -> None:
        self.model = tfmot.sparsity.keras.strip_pruning(self.model)

    def apply_pruning(self) -> None:
        """Apply all steps of pruning at once."""
        self._init_for_pruning()
        self._train_pruning()
        self._strip_pruning()
