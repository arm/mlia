# Copyright 2021, Arm Ltd.
"""Frequent use cases."""
import pandas as pd
from mlia.config import IPConfiguration
from mlia.config import TFLiteModel
from mlia.metrics import PerformanceMetrics
from mlia.optimizations.common import Optimizer
from mlia.performance import collect_performance_metrics
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import save_tflite_model
from mlia.utils.tflite_metrics import get_gzipped_file_size
from tensorflow.lite.python.interpreter import Interpreter


def get_metrics_and_size(model: TFLiteModel, device: IPConfiguration) -> pd.DataFrame:
    """Return a dataframe filled with performance metrics and model size."""
    metrics = collect_performance_metrics(model, device)
    size = get_gzipped_file_size(model.model_path)

    df = metrics.to_df()
    df["compressed file size"] = size
    df.columns = PerformanceMetrics.row_names + ["Compressed file size (bytes)"]

    return df


def optimize_and_compare(optimizer: Optimizer, device: IPConfiguration) -> pd.DataFrame:
    """Optimize model, return table that compares the original and optimized version."""
    tflite_model = convert_to_tflite(optimizer.get_model(), True)
    original = get_metrics_and_size(
        TFLiteModel(save_tflite_model(tflite_model)), device
    )

    tflite_optimized_model = optimize_model(optimizer)
    optimized = get_metrics_and_size(
        TFLiteModel(save_tflite_model(tflite_optimized_model)), device
    )

    # calculate percentage differences
    difference = 100 - (optimized / original * 100)

    results = original.append(optimized).append(difference)
    results = results.T
    results.columns = ["Original", "Optimized", "Improvement (%)"]

    results["Original"] = results["Original"].map("{:,.0f}".format)
    results["Optimized"] = results["Optimized"].map("{:,.0f}".format)
    results["Improvement (%)"] = results["Improvement (%)"].fillna(0)
    results["Improvement (%)"] = results["Improvement (%)"].map("{:.2f}".format)

    return results


def optimize_model(optimizer: Optimizer) -> Interpreter:
    """Optimize model and return the result."""
    optimizer.apply_optimization()
    optimized_keras_model = optimizer.get_model()
    optimized_tflite_model = convert_to_tflite(optimized_keras_model, True)

    return optimized_tflite_model
