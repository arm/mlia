# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contains class RewritingOptimizer to replace a subgraph/layer of a model."""
from __future__ import annotations

import logging
import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.core.errors import ConfigurationError
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import Table
from mlia.nn.common import Optimizer
from mlia.nn.common import OptimizerConfiguration
from mlia.nn.rewrite.core.train import train
from mlia.nn.rewrite.core.train import TrainingParameters
from mlia.nn.rewrite.library.fc_layer import get_keras_model as fc_rewrite
from mlia.nn.rewrite.library.fc_sparsity24_layer import (
    get_keras_model as fc_rewrite_sparsity24,
)
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.utils.registry import Registry


logger = logging.getLogger(__name__)
RewriteCallable = Callable[[Any, Any], keras.Model]


class Rewrite(ABC):
    """Abstract class for rewrite logic to be used by RewritingOptimizer."""

    def __init__(self, name: str, rewrite_fn: RewriteCallable):
        """Initialize a Rewrite instance with a given name and an optional function."""
        self.name = name
        self.function = rewrite_fn

    def __call__(self, input_shape: Any, output_shape: Any) -> keras.Model:
        """Perform the rewrite operation using the configured function."""
        try:
            return self.function(input_shape, output_shape)
        except Exception as ex:
            raise RuntimeError(f"Rewrite '{self.name}' failed.") from ex

    @abstractmethod
    def quantize(self, model: keras.Model) -> keras.Model:
        """Return a quantized model if required."""

    @abstractmethod
    def training_callbacks(self) -> list:
        """Return default rewrite callbacks."""

    @abstractmethod
    def post_process(self, model: keras.Model) -> keras.Model:
        """Return default post-processing rewrite options."""


class QATRewrite(Rewrite):
    """Logic for rewrites requiring quantization-aware training."""

    def pruning_preserved_quantization(
        self,
        model: keras.Model,
    ) -> keras.Model:
        """Apply pruning-preserved quantization training to a given model."""
        model = tfmot.quantization.keras.quantize_annotate_model(model)
        model = tfmot.quantization.keras.quantize_apply(
            model,
            tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme(),
        )

        return model


class FullyConnectedRewrite(Rewrite):
    """Graph rewrite logic for fully-connected rewrite."""

    def quantize(self, model: keras.Model) -> keras.Model:
        """Return a quantized model if required."""
        model = tfmot.quantization.keras.quantize_model(model)
        return model

    def training_callbacks(self) -> list:
        """Return default rewrite callbacks."""
        return []

    def post_process(self, model: keras.Model) -> keras.Model:
        """Return default post-processing rewrite options."""
        return model


class Sparsity24Rewrite(QATRewrite):
    """Graph rewrite logic for fully-connected-sparsity24 rewrite."""

    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep

    strip_pruning_wrapper = staticmethod(tfmot.sparsity.keras.strip_pruning)

    def quantize(self, model: keras.Model) -> keras.Model:
        """Skip quantization when using pruning rewrite."""
        return model

    def training_callbacks(self) -> list:
        """Return pruning-specific rewrite callback."""
        return [self.pruning_callback()]

    def post_process(self, model: keras.Model) -> keras.Model:
        """Pruning-specific post-processing rewrite options."""
        return self.strip_pruning_wrapper(model)


class RewriteRegistry(Registry[Rewrite]):
    """Registry rewrite functions."""

    def __init__(self, rewrites: list[Rewrite] | None = None):
        """Set up a rewrite registry.

        Can optionally initialise with name->function pairs
        to be automatically loaded on demand
        """
        super().__init__()
        if rewrites:
            for rewrite in rewrites:
                self.register_rewrite(rewrite)

    def register_rewrite(self, rewrite: Rewrite) -> bool:
        """Register a rewrite."""
        return super().register(rewrite.name, rewrite)


@dataclass
class RewriteConfiguration(OptimizerConfiguration):
    """Rewrite configuration."""

    optimization_target: str
    layers_to_optimize: list[str] | None = None
    dataset: Path | None = None
    train_params: TrainingParameters = TrainingParameters()

    def __str__(self) -> str:
        """Return string representation of the configuration."""
        return f"rewrite: {self.optimization_target}"


class RewritingOptimizer(Optimizer):
    """RewritingOptimizer class for basic rewrite flow."""

    registry = RewriteRegistry(
        [
            FullyConnectedRewrite("fully-connected", fc_rewrite),
            Sparsity24Rewrite("fully-connected-sparsity24", fc_rewrite_sparsity24),
            FullyConnectedRewrite("fully-connected-clustering", fc_rewrite),
        ]
    )

    def __init__(
        self, tflite_model_path: Path, optimizer_configuration: RewriteConfiguration
    ):
        """Init RewritingOptimizer instance."""
        self.model = TFLiteModel(tflite_model_path)
        self.model_path = tflite_model_path
        self.optimizer_configuration = optimizer_configuration

    @classmethod
    def builtin_rewrite_names(cls) -> list:
        """Return all registered rewrite names."""
        return cls.registry.names()

    def apply_optimization(self) -> None:  # pylint: disable=too-many-locals
        """Apply the rewrite flow."""
        rewrite = RewritingOptimizer.registry.items[
            self.optimizer_configuration.optimization_target
        ]
        is_qat = isinstance(rewrite, QATRewrite)
        use_unmodified_model = True
        tflite_model = self.model.model_path
        tfrecord = str(self.optimizer_configuration.dataset)

        tmp_dir = tempfile.mkdtemp()
        tmp_output = Path(tmp_dir, "output.tflite")

        if not self.optimizer_configuration.layers_to_optimize:
            raise ConfigurationError(
                "Input and output tensor names need to be set for rewrite."
            )

        orig_vs_repl_stats, total_stats = train(
            source_model=tflite_model,
            unmodified_model=tflite_model if use_unmodified_model else None,
            output_model=str(tmp_output),
            input_tfrec=str(tfrecord),
            rewrite=rewrite,
            input_tensors=[self.optimizer_configuration.layers_to_optimize[0]],
            output_tensors=[self.optimizer_configuration.layers_to_optimize[1]],
            is_qat=is_qat,
            train_params=self.optimizer_configuration.train_params,
        )

        if orig_vs_repl_stats:
            model_stats: list = []
            cp_param = self.optimizer_configuration.train_params.checkpoint_at
            checkpoints = (
                [
                    "At checkpoint " + str(checkpoint) + " steps"
                    for checkpoint in cp_param
                ]
                if cp_param
                else []
            )
            checkpoints.append("All Steps")
            for checkpoint, orig_vs_repl_stat in zip(checkpoints, orig_vs_repl_stats):
                model_stats.append(
                    ["Replaced sub-graph: " + checkpoint]
                    + [f"{stat:.3f}" for stat in orig_vs_repl_stat]
                )
            total = ["Total model"] + [f"{stat:.3f}" for stat in total_stats]
            notes = (
                "These metrics show the difference between original model\n"
                "and the model optimized by the rewrite. The models are\n"
                "compared at two positions: directly after the replaced\n"
                "sub-graph and at the model output.\n"
                "MAE = Mean Absolute Error\n"
                "NRMSE = Normalized Root Mean Square Error"
            )

            table = Table(
                columns=[
                    Column(
                        "Original vs. Optimized",
                        alias="metric",
                        fmt=Format(wrap_width=40),
                    ),
                    Column("MAE", alias="value", fmt=Format(wrap_width=15)),
                    Column("NRMSE", alias="value", fmt=Format(wrap_width=15)),
                ],
                rows=[*model_stats, total],
                name="Rewrite performance metrics",
                alias="rewrite_performance_metrics",
                notes=notes,
            )
            logger.info(table.to_plain_text(show_title=True))
        self.model = TFLiteModel(tmp_output)

    def get_model(self) -> TFLiteModel:
        """Return optimized model."""
        return self.model

    def optimization_config(self) -> str:
        """Optimization configurations."""
        return str(self.optimizer_configuration)
