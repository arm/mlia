# Copyright 2021, Arm Ltd.
"""Frequent use cases."""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import tensorflow as tf
from mlia.config import IPConfiguration
from mlia.config import TFLiteModel
from mlia.metrics import PerformanceMetrics
from mlia.optimizations.common import Optimizer
from mlia.performance import collect_performance_metrics
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import save_keras_model
from mlia.utils.general import save_tflite_model
from mlia.utils.tflite_metrics import get_gzipped_file_size

LOGGER = logging.getLogger("mlia.performance")


def get_metrics_and_size(
    model: TFLiteModel,
    device: IPConfiguration,
    working_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Return a dataframe filled with performance metrics and model size."""
    metrics = collect_performance_metrics(model, device, working_dir)
    size = get_gzipped_file_size(model.model_path)

    df = metrics.to_df()
    df["compressed file size"] = size
    df.columns = PerformanceMetrics.row_names + ["Compressed file size (bytes)"]

    return df


def optimize_and_compare(
    optimizer: Optimizer, device: IPConfiguration, working_dir: Optional[str] = None
) -> pd.DataFrame:
    """Optimize model, return table that compares the original and optimized version."""
    models_path = Path(working_dir) if working_dir else Path.cwd()

    LOGGER.info(
        """Original model:
"""
    )
    tflite_model = convert_to_tflite(optimizer.get_model(), True)

    temp_base_tflite_model_path = models_path / "original_model.tflite"
    temp_opt_keras_model_path = models_path / "optimized_model.h5"
    temp_opt_tflite_model_path = models_path / "optimized_model.tflite"
    save_tflite_model(tflite_model, temp_base_tflite_model_path)

    original = get_metrics_and_size(TFLiteModel(temp_base_tflite_model_path), device)

    LOGGER.info(
        """
Optimized model:
"""
    )
    keras_optimized_model = optimize_model(optimizer)
    save_keras_model(keras_optimized_model, temp_opt_keras_model_path)
    tflite_optimized_model = convert_to_tflite(keras_optimized_model, True)
    save_tflite_model(tflite_optimized_model, temp_opt_tflite_model_path)
    optimized = get_metrics_and_size(TFLiteModel(temp_opt_tflite_model_path), device)

    # calculate percentage differences
    difference = 100 - (optimized / original * 100)

    results = original.append(optimized).append(difference)
    results = results.T
    results.columns = ["Original", "Optimized", "Improvement (%)"]
    results["Original"].iloc[:5] /= 1024.0
    results["Original"].iloc[:5] = results["Original"].iloc[:5].map("{:,.2f}".format)
    results["Original"].iloc[5:] = results["Original"].iloc[5:].map("{:,.0f}".format)
    results["Optimized"].iloc[:5] /= 1024.0
    results["Optimized"].iloc[:5] = results["Optimized"].iloc[:5].map("{:,.2f}".format)
    results["Optimized"].iloc[5:] = results["Optimized"].iloc[5:].map("{:,.0f}".format)
    results["Improvement (%)"] = results["Improvement (%)"].fillna(0)

    return results


def optimize_model(optimizer: Optimizer) -> tf.keras.Model:
    """Optimize model and return the result."""
    optimizer.apply_optimization()
    optimized_keras_model = optimizer.get_model()

    return optimized_keras_model
