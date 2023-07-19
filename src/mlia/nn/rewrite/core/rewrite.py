# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contains class Rewriter to replace a subgraph/layer of a model."""
from __future__ import annotations

import importlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError
from mlia.nn.common import Optimizer
from mlia.nn.common import OptimizerConfiguration
from mlia.nn.rewrite.core.train import train
from mlia.nn.rewrite.core.train import TrainingParameters
from mlia.nn.tensorflow.config import TFLiteModel


logger = logging.getLogger(__name__)


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


class Rewriter(Optimizer):
    """Rewriter class for basic rewrite flow."""

    def __init__(
        self, tflite_model_path: Path, optimizer_configuration: RewriteConfiguration
    ):
        """Init Rewriter instance."""
        self.model = TFLiteModel(tflite_model_path)
        self.model_path = tflite_model_path
        self.optimizer_configuration = optimizer_configuration

    def apply_optimization(self) -> None:
        """Apply the rewrite flow."""

        def get_function(arg: str) -> Any:
            module_name = ".".join(arg.split(".")[:-1])
            fn_name = arg.split(".")[-1]
            module = importlib.import_module(module_name)
            return getattr(module, fn_name)

        if self.optimizer_configuration.optimization_target == "fully_connected":
            replace_function = "mlia.nn.rewrite.library.fc_layer.get_keras_model"
        else:
            raise ConfigurationError(
                "Only fully_connected replacement is supported in rewrite module."
            )

        replace_fn = get_function(replace_function)

        use_unmodified_model = True
        tflite_model = self.model.model_path
        tfrecord = str(self.optimizer_configuration.dataset)

        tmp_dir = tempfile.mkdtemp()
        tmp_output = Path(tmp_dir, "output.tflite")

        if not self.optimizer_configuration.layers_to_optimize:
            raise ConfigurationError(
                "Input and output tensor names need to be set for rewrite."
            )
        result = train(
            source_model=tflite_model,
            unmodified_model=tflite_model if use_unmodified_model else None,
            output_model=str(tmp_output),
            input_tfrec=str(tfrecord),
            replace_fn=replace_fn,
            input_tensors=[self.optimizer_configuration.layers_to_optimize[0]],
            output_tensors=[self.optimizer_configuration.layers_to_optimize[1]],
            train_params=self.optimizer_configuration.train_params,
        )

        self.model = TFLiteModel(tmp_output)

        if result:
            stats_as_str = ", ".join(str(stats) for stats in result)
            logger.info(
                "The MAE and NRMSE between original and replacement [%s]",
                stats_as_str,
            )

    def get_model(self) -> TFLiteModel:
        """Return optimized model."""
        return self.model

    def optimization_config(self) -> str:
        """Optimization configurations."""
        return str(self.optimizer_configuration)
