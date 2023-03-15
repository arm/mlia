# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contains class Rewriter to replace a subgraph/layer of a model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlia.nn.common import Optimizer
from mlia.nn.common import OptimizerConfiguration
from mlia.nn.tensorflow.config import TFLiteModel


@dataclass
class RewriteConfiguration(OptimizerConfiguration):
    """Rewrite configuration."""

    optimization_target: str
    layers_to_optimize: list[str] | None = None
    dataset: Path | None = None

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
        self.optimizer_configuration = optimizer_configuration

    def apply_optimization(self) -> None:
        """Apply the rewrite flow."""

    def get_model(self) -> TFLiteModel:
        """Return optimized model."""
        return self.model

    def optimization_config(self) -> str:
        """Optimization configurations."""
        return str(self.optimizer_configuration)
