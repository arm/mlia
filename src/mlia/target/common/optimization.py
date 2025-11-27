# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collector support for performance optimizations."""
from __future__ import annotations

import logging
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable

from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.performance import estimate_performance
from mlia.core.performance import P
from mlia.core.performance import PerformanceEstimator
from mlia.nn.rewrite.core.train import AUGMENTATION_PRESETS
from mlia.nn.select import get_optimizer
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.config import get_keras_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.utils import save_keras_model
from mlia.nn.tensorflow.utils import save_tflite_model
from mlia.target.config import TargetProfile
from mlia.utils.types import is_list_of

logger = logging.getLogger(__name__)


class OptimizingDataCollector(ContextAwareDataCollector):
    """Collect performance metrics for the optimizations."""

    def __init__(
        self,
        model: Path,
        target_config: TargetProfile,
        backends: list[str] | None = None,
    ) -> None:
        """Init performance optimizations data collector."""
        self.model = model
        self.target = target_config
        self.backends = backends

    def collect_data(self) -> DataItem:
        """Collect performance metrics for the optimizations."""
        logger.info("Estimate performance ...")

        optimizations = self._get_optimization_settings(self.context)

        rewrite_parameters = self._get_rewrite_settings(self.context)

        if not optimizations or optimizations == [[]]:
            raise FunctionalityNotSupportedError(
                reason="No optimization targets provided",
                description="Unable to estimate model optimizations impact",
            )

        opt_settings = self._parse_optimization_params(optimizations)

        optimization_types = {
            setting.optimization_type for opt in opt_settings for setting in opt
        }

        if optimization_types != {"rewrite"}:
            try:
                model = get_keras_model(self.model, self.context)
            except NotImplementedError as err:
                raise FunctionalityNotSupportedError(
                    reason=f"{self.model} is not a Keras model and "
                    "could not be converted to a Keras model",
                    description="Unable to run model optimizations",
                ) from err
        else:
            model = self.model  # type: ignore

        optimizers: list[Callable] = [
            partial(self.optimize_model, opts, rewrite_parameters)
            for opts in opt_settings
        ]

        return self.optimize_and_estimate_performance(model, optimizers, opt_settings)

    def optimize_model(
        self,
        opt_settings: list[OptimizationSettings],
        rewrite_parameters: dict,
        model: KerasModel | TFLiteModel | Path,
    ) -> Any:
        """Run optimization."""
        optimizer = get_optimizer(
            model, opt_settings, rewrite_parameters=rewrite_parameters
        )
        opts_as_str = ", ".join(str(opt) for opt in opt_settings)
        logger.info("Applying model optimizations - [%s]", opts_as_str)
        optimizer.apply_optimization()

        model = optimizer.get_model()  # type: ignore

        if isinstance(model, Path):
            return model

        if isinstance(model, TFLiteModel):
            model_path = self.context.get_model_path("optimized_model.tflite")
            with open(model.model_path, "rb") as file_handle:
                model_data = bytearray(file_handle.read())
            save_tflite_model(model_data, model_path)
            return TFLiteModel(model_path)

        model_path = self.context.get_model_path("optimized_model.h5")
        save_keras_model(model, model_path)
        return KerasModel(model_path)

    def _get_optimization_settings(self, context: Context) -> list[list[dict]]:
        """Get optimization settings."""
        return self.get_parameter(  # type: ignore
            OptimizingDataCollector.name(),
            "optimizations",
            expected_type=list,
            expected=False,
            context=context,
        )

    def _get_rewrite_settings(self, context: Context) -> list[dict]:
        """Get optimization settings."""
        return self.get_parameter(  # type: ignore
            OptimizingDataCollector.name(),
            "rewrite_parameters",
            expected_type=dict,
            expected=False,
            context=context,
        )

    @staticmethod
    def _parse_optimization_params(
        optimizations: list[list[dict]],
    ) -> list[list[OptimizationSettings]]:
        """Parse optimization parameters."""
        if not is_list_of(optimizations, list):
            raise TypeError("Optimization parameters expected to be a list.")

        return [
            [
                OptimizationSettings(
                    item.get("optimization_type"),  # type: ignore
                    item.get("optimization_target"),  # type: ignore
                    item.get("layers_to_optimize"),
                    item.get("dataset"),
                )
                for item in opt_configuration
            ]
            for opt_configuration in optimizations
        ]

    def optimize_and_estimate_performance(
        self,
        model: KerasModel | Path,
        optimizers: list[Callable],
        _: list[list[OptimizationSettings]],
    ) -> DataItem:
        """Run optimizers and estimate perfomance on the results."""
        for optimizer in optimizers:
            optimizer(model)

        return {}

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "common_optimizations"


class OptimizingPerformaceDataCollector(OptimizingDataCollector):
    """Collect performance metrics for the optimizations."""

    @abstractmethod
    def create_estimator(self) -> PerformanceEstimator:
        """Create a PerformanceEstimator, to be overridden in subclasses."""

    @abstractmethod
    def create_optimization_performance_metrics(
        self, original_metrics: P, optimizations_perf_metrics: list[P]
    ) -> Any:
        """Create an optimization metrics object."""

    def optimize_and_estimate_performance(
        self,
        model: KerasModel | Path,
        optimizers: list[Callable],
        opt_settings: list[list[OptimizationSettings]],
    ) -> Any:
        """Run optimizers and estimate perfomance on the results."""
        estimator = self.create_estimator()

        original_metrics, *optimized_metrics = estimate_performance(
            model, estimator, optimizers
        )

        return self.create_optimization_performance_metrics(
            original_metrics,
            list(zip(opt_settings, optimized_metrics)),
        )


_DEFAULT_OPTIMIZATION_TARGETS = [
    {
        "optimization_type": "pruning",
        "optimization_target": 0.5,
        "layers_to_optimize": None,
    },
    {
        "optimization_type": "clustering",
        "optimization_target": 32,
        "layers_to_optimize": None,
    },
]


def parse_augmentations(
    augmentations: dict | str | None,
) -> tuple[float | None, float | None]:
    """Parse augmentations from optimization-profile and return a valid tuple."""
    if isinstance(augmentations, str):
        match_augmentation = AUGMENTATION_PRESETS.get(augmentations)
        if not match_augmentation:
            match_augmentation = AUGMENTATION_PRESETS["none"]
        return match_augmentation
    if isinstance(augmentations, dict):
        augmentation_keys_test_for_valid = list(augmentations.keys())
        augmentation_keys_test_for_float = list(augmentations.keys())
        valid_keys = ["mixup_strength", "gaussian_strength"]
        tuple_to_return = []
        for valid_key in valid_keys.copy():
            if augmentations.get(valid_key) is not None:
                del augmentation_keys_test_for_valid[
                    augmentation_keys_test_for_valid.index(valid_key)
                ]
                if isinstance(augmentations.get(valid_key), float):
                    tuple_to_return.append(augmentations[valid_key])
                    del augmentation_keys_test_for_float[
                        augmentation_keys_test_for_float.index(valid_key)
                    ]
                else:
                    tuple_to_return.append(None)
            else:
                tuple_to_return.append(None)
        if len(augmentation_keys_test_for_valid) > 0:
            logger.warning(
                "Warning! Expected augmentation parameters to be 'gaussian_strength' "
                "and/or 'mixup_strength' got %s. "
                "Removing invalid augmentations",
                str(list(augmentations.keys())),
            )
        elif len(augmentation_keys_test_for_float) > 0:
            logger.warning(
                "Warning! Not all augmentation parameters were floats, "
                "removing non-float augmentations"
            )
        return (tuple_to_return[0], tuple_to_return[1])
    return AUGMENTATION_PRESETS["none"]


def add_common_optimization_params(  # pylint: disable=too-many-branches
    advisor_parameters: dict, extra_args: dict
) -> None:
    """Add common optimization parameters."""
    optimization_targets = extra_args.get("optimization_targets")
    if not optimization_targets:
        optimization_targets = _DEFAULT_OPTIMIZATION_TARGETS

    if not is_list_of(optimization_targets, dict):
        raise TypeError("Optimization targets value has wrong format.")

    training_parameters = None
    rewrite_specific_parameters = None

    optimization_parameters = extra_args.get("optimization_profile")
    if optimization_parameters:  # pylint: disable=too-many-nested-blocks
        if not isinstance(optimization_parameters, dict):
            raise TypeError("Optimization Parameter values has wrong format.")

        if optimization_parameters.get("rewrite"):
            rewrite_params = optimization_parameters["rewrite"]
            training_parameters = rewrite_params.get("training_parameters")
            if training_parameters:
                training_parameters["augmentations"] = parse_augmentations(
                    training_parameters.get("augmentations")
                )

            optimization_target = optimization_targets[0]["optimization_target"]
            rewrite_specific_parameters = rewrite_params.get(optimization_target)
    advisor_parameters.update(
        {
            "common_optimizations": {
                "optimizations": [optimization_targets],
                "rewrite_parameters": {
                    "train_params": training_parameters,
                    "rewrite_specific_params": rewrite_specific_parameters,
                },
            },
        }
    )
