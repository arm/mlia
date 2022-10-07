# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Contains class TFLiteMetrics to calculate metrics from a TensorFlow Lite file.

These metrics include:
* Sparsity (per layer and overall)
* Unique weights (clusters) (per layer)
* gzip compression ratio
"""
from __future__ import annotations

import os
import typing
from enum import Enum
from pprint import pprint
from typing import Any

import numpy as np
import tensorflow as tf

DEFAULT_IGNORE_LIST = [
    "relu",
    "pooling",
    "reshape",
    "identity",
    "input",
    "add",
    "flatten",
    "StatefulPartitionedCall",
    "bias",
]


@typing.no_type_check
def calculate_num_unique_weights(weights: np.ndarray) -> int:
    """Calculate the number of unique weights in the given weights."""
    # Types need to be ignored for this function call because
    # np.unique does not have type annotation while the
    # current context does.
    num_unique_weights = len(np.unique(weights))
    return num_unique_weights


def calculate_num_unique_weights_per_axis(weights: np.ndarray, axis: int) -> list[int]:
    """Calculate unique weights per quantization axis."""
    # Make quantized dimension the first dimension
    weights_trans = np.swapaxes(weights, 0, axis)
    num_uniques_weights = [
        calculate_num_unique_weights(weights_trans[i])
        for i in range(weights_trans.shape[0])
    ]
    assert num_uniques_weights
    return num_uniques_weights


class SparsityAccumulator:
    """Helper class to accumulate sparsity over several layers."""

    def __init__(self) -> None:
        """Create an empty accumulator."""
        self.total_non_zero_weights: int = 0
        self.total_weights: int = 0

    def __call__(self, weights: np.ndarray) -> None:
        """Update the accumulator with the given weights."""
        non_zero_weights = np.count_nonzero(weights)
        self.total_non_zero_weights += non_zero_weights
        self.total_weights += weights.size

    def sparsity(self) -> float:
        """Calculate the sparsity for all added weights."""
        return 1.0 - self.total_non_zero_weights / float(self.total_weights)


def calculate_sparsity(
    weights: np.ndarray, accumulator: SparsityAccumulator | None = None
) -> float:
    """
    Calculate the sparsity for the given weights.

    If the accumulator is passed, it is updated as well.
    """
    non_zero_weights = np.count_nonzero(weights)
    sparsity = 1.0 - float(non_zero_weights) / float(weights.size)
    if accumulator is not None:
        accumulator(weights)
    return sparsity


class ReportClusterMode(Enum):
    """Specifies the way cluster values are aggregated and reported."""

    NUM_CLUSTERS_HISTOGRAM = (
        "A histogram of the number of clusters per axis. "
        "I.e. the number of clusters is the index of the list (the bin) and "
        "the value is the number of axes that have this number of clusters. "
        "The first bin is 1."
    )
    NUM_CLUSTERS_PER_AXIS = "Number of clusters (unique weights) per axis."
    NUM_CLUSTERS_MIN_MAX = "Min/max number of clusters over all axes."


class TFLiteMetrics:
    """Helper class to calculate metrics from a TensorFlow Lite file.

    Metrics include:
    * sparsity (per-layer and overall)
    * number of unique weights (clusters) per layer
    * File compression via gzip
    """

    def __init__(self, tflite_file: str, ignore_list: list[str] | None = None) -> None:
        """Load the TensorFlow Lite file and filter layers."""
        self.tflite_file = tflite_file
        if ignore_list is None:
            ignore_list = DEFAULT_IGNORE_LIST
        self.ignore_list = [ignore.casefold() for ignore in ignore_list]
        # Initialize the TensorFlow Lite interpreter with the model file
        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_file, experimental_preserve_all_tensors=True
        )
        self.interpreter.allocate_tensors()
        self.details: dict = {}

        def ignore(details: dict) -> bool:
            name = details["name"].casefold()
            if not name:
                return True
            for to_ignore in self.ignore_list:
                if to_ignore in name:
                    return True
            return False

        self.filtered_details = {
            details["name"]: details
            for details in self.interpreter.get_tensor_details()
            if not ignore(details)
        }

    def get_tensor(self, details: dict) -> Any:
        """Return the weights/tensor specified in the given details map."""
        return self.interpreter.tensor(details["index"])()

    def sparsity_per_layer(self) -> dict:
        """Return a dict of layer name and sparsity value."""
        sparsity = {
            name: calculate_sparsity(self.get_tensor(details))
            for name, details in self.filtered_details.items()
        }
        return sparsity

    def sparsity_overall(self) -> float:
        """Return an instance of SparsityAccumulator for the filtered layers."""
        acc = SparsityAccumulator()
        for details in self.filtered_details.values():
            acc(self.get_tensor(details))
        return acc.sparsity()

    def calc_num_clusters_per_axis(self, details: dict) -> list[int]:
        """Calculate number of clusters per axis."""
        quant_params = details["quantization_parameters"]
        per_axis = len(quant_params["zero_points"]) > 1
        if per_axis:
            # Calculate unique weights along quantization axis
            axis = quant_params["quantized_dimension"]
            return calculate_num_unique_weights_per_axis(self.get_tensor(details), axis)

        # Calculate unique weights over all axes/dimensions
        return [calculate_num_unique_weights(self.get_tensor(details))]

    def num_unique_weights(self, mode: ReportClusterMode) -> dict:
        """Return a dict of layer name and number of unique weights."""
        aggregation_func = None
        if mode == ReportClusterMode.NUM_CLUSTERS_PER_AXIS:
            aggregation_func = self.calc_num_clusters_per_axis
        elif mode == ReportClusterMode.NUM_CLUSTERS_MIN_MAX:

            def cluster_min_max(details: dict) -> list[int]:
                num_clusters = self.calc_num_clusters_per_axis(details)
                return [min(num_clusters), max(num_clusters)]

            aggregation_func = cluster_min_max
        elif mode == ReportClusterMode.NUM_CLUSTERS_HISTOGRAM:

            def cluster_hist(details: dict) -> list[int]:
                num_clusters = self.calc_num_clusters_per_axis(details)
                max_num = max(num_clusters)
                hist = [0] * (max_num)
                for num in num_clusters:
                    idx = num - 1
                    hist[idx] += 1
                return hist

            aggregation_func = cluster_hist
        else:
            raise NotImplementedError(f"ReportClusterMode '{mode}' not implemented.")
        uniques = {
            name: aggregation_func(details)
            for name, details in self.filtered_details.items()
        }
        return uniques

    @staticmethod
    def _prettify_name(name: str) -> str:
        if name.startswith("model"):
            return name.split("/", 1)[1]
        return name

    @typing.no_type_check
    def summary(
        self,
        report_sparsity: bool,
        report_cluster_mode: ReportClusterMode = None,
        max_num_clusters: int = 32,
        verbose: bool = False,
    ) -> None:
        """Print a summary of all the model information."""
        print(f"Model file: {self.tflite_file}")
        print("#" * 80)
        print(" " * 28 + "### TENSORFLOW LITE SUMMARY ###")
        print(f"File: {os.path.abspath(self.tflite_file)}")
        print("Input(s):")
        self._print_in_outs(self.interpreter.get_input_details(), verbose)
        print("Output(s):")
        self._print_in_outs(self.interpreter.get_output_details(), verbose)
        print()
        header = ["Layer", "Index", "Type", "Num weights"]
        if report_sparsity:
            header.append("Sparsity")
        rows = []
        sparsity_accumulator = SparsityAccumulator()
        for details in self.filtered_details.values():
            name = details["name"]
            weights = self.get_tensor(details)
            row = [
                self._prettify_name(name),
                details["index"],
                weights.dtype,
                weights.size,
            ]
            if report_sparsity:
                sparsity = calculate_sparsity(weights, sparsity_accumulator)
                row.append(f"{sparsity:.2f}")
            rows.append(row)
            if verbose:
                # Print cluster centroids
                print(f"{name} cluster centroids:")
                # Types need to be ignored for this function call because
                # np.unique does not have type annotation while the
                # current context does.
                pprint(np.unique(weights))
        # Add summary/overall values
        empty_row = ["" for _ in range(len(header))]
        summary_row = empty_row
        summary_row[header.index("Layer")] = "=> OVERALL"
        summary_row[header.index("Num weights")] = str(
            sparsity_accumulator.total_weights
        )
        if report_sparsity:
            summary_row[
                header.index("Sparsity")
            ] = f"{sparsity_accumulator.sparsity():.2f}"
        rows.append(summary_row)
        # Report detailed cluster info
        if report_cluster_mode is not None:
            print()
            self._print_cluster_details(report_cluster_mode, max_num_clusters)
        print("#" * 80)

    def _print_cluster_details(
        self, report_cluster_mode: ReportClusterMode, max_num_clusters: int
    ) -> None:
        print(f"{report_cluster_mode.name}:\n{report_cluster_mode.value}")
        num_clusters = self.num_unique_weights(report_cluster_mode)
        if (
            report_cluster_mode == ReportClusterMode.NUM_CLUSTERS_HISTOGRAM
            and max_num_clusters > 0
        ):
            # Only show cluster histogram if there are not more than
            # max_num_clusters. This is a workaround for not showing a huge
            # histogram for unclustered layers.
            for name, value in num_clusters.items():
                if len(value) > max_num_clusters:
                    num_clusters[name] = f"More than {max_num_clusters} unique values."
        for name, nums in num_clusters.items():
            print(f"- {self._prettify_name(name)}: {nums}")

    @staticmethod
    def _print_in_outs(ios: list[dict], verbose: bool = False) -> None:
        for item in ios:
            if verbose:
                pprint(item)
            else:
                print(
                    f"- {item['name']} ({np.dtype(item['dtype']).name}): "
                    f"{item['shape']}"
                )
