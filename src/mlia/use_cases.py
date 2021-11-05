# Copyright 2021, Arm Ltd.
"""Frequent use cases."""
import logging
from typing import Tuple

import pandas as pd
import tensorflow as tf
from mlia.config import Context
from mlia.config import IPConfiguration
from mlia.config import TFLiteModel
from mlia.metrics import PerformanceMetrics
from mlia.optimizations.common import Optimizer
from mlia.performance import collect_performance_metrics
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import save_keras_model
from mlia.utils.general import save_tflite_model

logger = logging.getLogger(__name__)


def optimize_and_compare(
    optimizer: Optimizer, device: IPConfiguration, ctx: Context
) -> Tuple[PerformanceMetrics, PerformanceMetrics]:
    """Optimize model, return perf metrics for the original and optimized version."""
    logger.info("Original model:\n")
    original_model = optimizer.get_model()
    original = _process_model(original_model, device, "original", False, ctx)

    logger.info("\nOptimized model:\n")
    optimized_model = optimize_model(optimizer)
    optimized = _process_model(optimized_model, device, "optimized", True, ctx)

    return (original, optimized)


def _process_model(
    keras_model: tf.keras.Model,
    device: IPConfiguration,
    prefix: str,
    save_model: bool,
    ctx: Context,
) -> PerformanceMetrics:
    """Convert and estimate performance for the model."""
    if save_model:
        keras_model_path = ctx.get_model_path(f"{prefix}_model.h5")
        save_keras_model(keras_model, keras_model_path)

    tflite_model = convert_to_tflite(keras_model, True)
    tflite_model_path = ctx.get_model_path(f"{prefix}_model.tflite")
    save_tflite_model(tflite_model, tflite_model_path)

    return collect_performance_metrics(TFLiteModel(tflite_model_path), device, ctx)


def compare_metrics(
    original: PerformanceMetrics, optimized: PerformanceMetrics
) -> pd.DataFrame:
    """Compare performance metrics."""
    original_df = original.in_kilobytes().to_df()
    optimized_df = optimized.in_kilobytes().to_df()

    # calculate percentage differences
    difference_df = (100 - (optimized_df / original_df * 100)).fillna(0)

    results = original_df.append(optimized_df).append(difference_df).T
    results.columns = ["Original", "Optimized", "Improvement (%)"]

    return results


def optimize_model(optimizer: Optimizer) -> tf.keras.Model:
    """Optimize model and return the result."""
    logger.info("Applying optimizations (%s) ...", optimizer.optimization_config())

    optimizer.apply_optimization()
    optimized_keras_model = optimizer.get_model()

    logger.info("Done")

    return optimized_keras_model
