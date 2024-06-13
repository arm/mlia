# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contains class RewritingOptimizer to replace a subgraph/layer of a model."""
from __future__ import annotations

import logging
import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from inspect import getfullargspec
from pathlib import Path
from statistics import fmean
from typing import Any
from typing import Callable
from typing import Generator

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_utils import (  # pylint: disable=no-name-in-module
    is_pruned_m_by_n,
)

from mlia.core.errors import ConfigurationError
from mlia.core.reporting import Column
from mlia.core.reporting import Format
from mlia.core.reporting import Table
from mlia.nn.common import Optimizer
from mlia.nn.common import OptimizerConfiguration
from mlia.nn.rewrite.core.train import train
from mlia.nn.rewrite.core.train import TrainingParameters
from mlia.nn.rewrite.library.clustering import conv2d_clustering_rewrite
from mlia.nn.rewrite.library.clustering import fc_clustering_rewrite
from mlia.nn.rewrite.library.layers import conv2d_rewrite
from mlia.nn.rewrite.library.layers import fc_rewrite
from mlia.nn.rewrite.library.sparsity import conv2d_sparsity_rewrite
from mlia.nn.rewrite.library.sparsity import conv2d_sparsity_unstructured_rewrite
from mlia.nn.rewrite.library.sparsity import fc_sparsity_rewrite
from mlia.nn.rewrite.library.sparsity import fc_sparsity_unstructured_rewrite
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.utils.registry import Registry

logger = logging.getLogger(__name__)
RewriteCallable = Callable[..., keras.Model]


class Rewrite(ABC):
    """Abstract class for rewrite logic to be used by RewritingOptimizer."""

    def __init__(
        self,
        name: str,
        rewrite_fn: RewriteCallable,
        rewrite_fn_extra_args: dict[str, Any] | None = None,
    ):
        """Initialize a Rewrite instance with a given name and an optional function."""
        self.name = name
        self.function = rewrite_fn
        self.function_extra_args = rewrite_fn_extra_args

    def __call__(
        self, input_shape: Any, output_shape: Any, **kwargs: Any
    ) -> keras.Model:
        """Perform the rewrite operation using the configured function."""
        try:
            if self.function_extra_args:
                return self.function(
                    input_shape, output_shape, **kwargs, **self.function_extra_args
                )

            return self.function(input_shape, output_shape, **kwargs)
        except TypeError as ex:
            expected_args = self.return_rewrite_func_args()
            if "input_shape" in expected_args:
                expected_args.remove("input_shape")
            if "output_shape" in expected_args:
                expected_args.remove("output_shape")
            raise KeyError(
                f"Found unexpected parameters for rewrite. Expected (sub)set "
                f"of {expected_args} found unexpected parameter(s) "
                f"{list(set(list(kwargs.keys())) - set(expected_args))}"
            ) from ex
        except Exception as ex:
            raise RuntimeError(f"Rewrite '{self.name}' failed.") from ex

    def quantize(self, model: keras.Model) -> keras.Model:
        """Return a quantized model if required."""
        return model

    def return_rewrite_func_args(self) -> list[str]:
        """Return the expected args of the rewrite function."""
        return getfullargspec(self.function).args

    @abstractmethod
    def training_callbacks(self) -> list:
        """Return rewrite callbacks."""

    @abstractmethod
    def post_process(self, model: keras.Model) -> keras.Model:
        """Return post-processing rewrite option."""

    @abstractmethod
    def check_optimization(self, model: keras.Model) -> bool:
        """Check if the optimization has produced the correct result."""


class GenericRewrite(Rewrite):
    """Rewrite class for generic rewrites e.g. fully-connected."""

    def quantize(self, model: keras.Model) -> keras.Model:
        """Return a quantized model if required."""
        return tfmot.quantization.keras.quantize_model(model)

    def training_callbacks(self) -> list:
        """Return default rewrite callbacks."""
        return []

    def post_process(self, model: keras.Model) -> keras.Model:
        """Return default post-processing rewrite option."""
        return model

    def check_optimization(self, model: keras.Model, **_: Any) -> bool:
        """Not needed here."""
        return True


class QuantizeAwareTrainingRewrite(Rewrite, ABC):
    """Abstract class for rewrites that perform QAT."""

    @abstractmethod
    def preserved_quantize(self, model: keras.Model) -> keras.Model:
        """Apply optimization-aware quantization to a given model."""
        return model

    def check_optimization_generator(
        self, model: keras.Model
    ) -> Generator[tuple[tf.Tensor, keras.layers.Layer], None, None]:
        """Loop for check_optimization function."""
        for layer in model.layers:
            for weight in layer.weights:
                if "kernel" in weight.name:
                    if (
                        "kernel_min" in weight.name
                        or "kernel_max" in weight.name
                        or "depthwise" in weight.name
                    ):
                        continue
                    yield weight, layer


class SparsityRewrite(QuantizeAwareTrainingRewrite):
    """Base rewrite class for sparsity rewrites."""

    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep

    strip_pruning_wrapper = staticmethod(tfmot.sparsity.keras.strip_pruning)

    def quantize(self, model: keras.Model) -> keras.Model:
        """Skip quantization when using sparsity rewrite."""
        return model

    def training_callbacks(self) -> list:
        """Return pruning-specific rewrite callback."""
        return [self.pruning_callback()]

    def post_process(self, model: keras.Model) -> keras.Model:
        """Pruning-specific post-processing rewrite option."""
        return self.strip_pruning_wrapper(model)

    def preserved_quantize(
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

    def check_optimization(self, model: keras.Model) -> bool:
        """Not needed here."""
        return True


class UnstructuredSparsityRewrite(SparsityRewrite):
    """
    Rewrite class for unstructured sparsity rewrite.

    e.g. fully-connected-unstructured-sparsity.
    """

    def check_optimization(
        self, model: keras.Model, final_sparsity: float = 0.5, **_: Any
    ) -> bool:
        """Not needed here."""
        found_sparsity_list = []
        num_dec_places = str(final_sparsity)[::-1].find(".")
        for weight, _ in self.check_optimization_generator(model=model):
            weight_np = weight.numpy()
            found_sparsity_list.append(
                round(np.count_nonzero(weight_np) / weight_np.size, num_dec_places)
            )
        if len(found_sparsity_list) == 0:
            logger.warning(
                "\nWARNING: Could not find any layers "
                "in rewrite that could be sparsely pruned"
            )
            return False
        found_sparsity = fmean(found_sparsity_list)
        if found_sparsity != final_sparsity:
            logger.warning(
                "\nWARNING: Found total sparsity of "
                "rewrite model: %.2f "
                "expected total sparsity to be: "
                "%.2f\n",
                found_sparsity,
                final_sparsity,
            )
            return False
        return True


class StructuredSparsityRewrite(SparsityRewrite):
    """Rewrite class for structured sparsity rewrite e.g. fully-connected-sparsity."""

    def check_optimization(
        self,
        model: keras.Model,
        sparsity_m: int = 2,
        sparsity_n: int = 4,
        **_: Any,
    ) -> bool:
        """Check if sparity has produced the correct result."""
        for weight, layer in self.check_optimization_generator(model=model):
            if not is_pruned_m_by_n(weight, m_by_n=(sparsity_m, sparsity_n)):
                logger.warning(
                    "\nWARNING: Could not find (%d, %d) sparsity, "
                    "in layer %s for weight %s \n",
                    sparsity_m,
                    sparsity_n,
                    layer.name,
                    weight.name,
                )
                return False
        return True


class ClusteringRewrite(QuantizeAwareTrainingRewrite):
    """Rewrite class for clustering rewrite e.g. fully-connected-clustering."""

    _strip_clustering_wrapper = staticmethod(tfmot.clustering.keras.strip_clustering)

    def preserved_quantize(self, model: keras.Model) -> keras.Model:
        """Apply clustering-preserved quantization to a given model."""
        quant_aware_model = tfmot.quantization.keras.quantize_annotate_model(model)
        cqat_model = tfmot.quantization.keras.quantize_apply(
            quant_aware_model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(),
        )
        return cqat_model

    def check_optimization(
        self, model: keras.Model, num_clusters: int = 2, **_: Any
    ) -> bool:
        """Check if clustering has produced the correct result."""
        for weight, layer in self.check_optimization_generator(model=model):
            number_of_found_clusters = len(np.unique(weight))
            if number_of_found_clusters != num_clusters:
                logger.warning(
                    "\nWARNING: Expected %d cluster(s), found %d "
                    "cluster(s) in layer %s for weight %s \n",
                    num_clusters,
                    number_of_found_clusters,
                    layer.name,
                    weight.name,
                )
                return False
        return True

    def training_callbacks(self) -> list:
        """Return default rewrite callbacks."""
        return []

    def post_process(self, model: keras.Model) -> keras.Model:
        """Clustering-specific post-processing rewrite option."""
        return self._strip_clustering_wrapper(model)


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
    rewrite_specific_params: dict | None = None

    def __str__(self) -> str:
        """Return string representation of the configuration."""
        return f"rewrite: {self.optimization_target}"


class RewritingOptimizer(Optimizer):
    """RewritingOptimizer class for basic rewrite flow."""

    registry = RewriteRegistry(
        [
            GenericRewrite("fully-connected", fc_rewrite),
            GenericRewrite("conv2d", conv2d_rewrite),
            GenericRewrite(
                "depthwise-separable-conv2d",
                conv2d_rewrite,
                {"layer_type": keras.layers.SeparableConv2D},
            ),
            StructuredSparsityRewrite("fully-connected-sparsity", fc_sparsity_rewrite),
            ClusteringRewrite("fully-connected-clustering", fc_clustering_rewrite),
            ClusteringRewrite("conv2d-clustering", conv2d_clustering_rewrite),
            StructuredSparsityRewrite("conv2d-sparsity", conv2d_sparsity_rewrite),
            UnstructuredSparsityRewrite(
                "conv2d-unstructured-sparsity", conv2d_sparsity_unstructured_rewrite
            ),
            UnstructuredSparsityRewrite(
                "fully-connected-unstructured-sparsity",
                fc_sparsity_unstructured_rewrite,
            ),
            ClusteringRewrite(
                "depthwise-separable-conv2d-clustering",
                conv2d_clustering_rewrite,
                {"layer_type": keras.layers.SeparableConv2D},
            ),
            StructuredSparsityRewrite(
                "depthwise-separable-conv2d-sparsity",
                conv2d_sparsity_rewrite,
                {"layer_type": keras.layers.SeparableConv2D},
            ),
            UnstructuredSparsityRewrite(
                "depthwise-separable-conv2d-unstructured-sparsity",
                conv2d_sparsity_unstructured_rewrite,
                {"layer_type": keras.layers.SeparableConv2D},
            ),
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
            is_qat=isinstance(rewrite, QuantizeAwareTrainingRewrite),
            input_tensors=[self.optimizer_configuration.layers_to_optimize[0]],
            output_tensors=[self.optimizer_configuration.layers_to_optimize[1]],
            train_params=self.optimizer_configuration.train_params,
            rewrite_specific_params=self.optimizer_configuration.rewrite_specific_params,  # pylint: disable=line-too-long
            detect_activation_function=(
                "activation" in rewrite.return_rewrite_func_args()
            ),
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
