# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Diff module: compare subgraph outputs."""
# pylint: disable=too-many-locals
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def dict_mean(mean_dict: dict) -> Any:
    """Return the mean of values in a given dict."""
    return np.mean(list(mean_dict.values()))


def add_total(name: str, key: str, values: list, totals: dict) -> None:
    """Append values to dict totals."""
    if key not in totals[name]:
        totals[name][key] = values
    else:
        totals[name][key] += values


def _handle_zeros_in_denominator(denominator: np.ndarray) -> np.ndarray:
    """Handle zeros in the denominator in nrmse to avoid dividing by zero(s)."""
    denominator[denominator == 0.0] = 1.0
    return denominator


def calc_nrmse(rmse: dict, dataset1_var: dict) -> dict:
    """Divide rmse by target standard deviation."""
    nrmse = {
        k: v / _handle_zeros_in_denominator(np.sqrt(dataset1_var[k]))
        for k, v in rmse.items()
    }
    return nrmse


def diff_stats(
    file1: str | Path, file2: str | Path, per_tensor_and_channel: bool = False
) -> tuple:
    """Compare the statistics of outputs between two subgraphs."""
    dataset1 = numpytf_read(file1)
    dataset2 = numpytf_read(file2)

    totals: dict = defaultdict(dict)

    # First iterate through dataset and calculate per-channel total for each tensor
    count = 0
    for data in dataset1:
        count += 1
        for key, val in data.items():
            value = val.numpy().astype(np.double)
            add_total("dataset1_total", key, value, totals)

    # Use this to calculate per-channel mean for each tensor
    def per_tensor_mean(name: str) -> dict:
        return {k: total / count for k, total in totals[name].items()}

    dataset1_mean = per_tensor_mean("dataset1_total")

    # Next iterate through both datasets and calculate per-channel total squared
    # error between them for each tensor and dataset1 variance for each tensor
    # using the mean from above
    for i, (ds1, ds2) in enumerate(zip(dataset1, dataset2)):
        assert ds1.keys() == ds2.keys(), (
            f"At input {i} the files have different sets of tensors.\n"
            f"{file1}: {', '.join(ds1.keys())}\n"
            f"{file2}: {', '.join(ds2.keys())}\n"
        )
        for key in ds1.keys():
            tensor1 = ds1[key].numpy().astype(np.double)
            tensor2 = ds2[key].numpy().astype(np.double)
            add_total("ae", key, abs(tensor1 - tensor2), totals)
            add_total("se", key, (tensor1 - tensor2) ** 2, totals)
            add_total(
                "dataset1_variance",
                key,
                (tensor1 - dataset1_mean[key]) ** 2,
                totals,
            )

    # Finally average over number of inputs to get the rmse and the dataset1 variance
    mae = per_tensor_mean("ae")
    mse = per_tensor_mean("se")
    rmse = {k: np.sqrt(v) for k, v in mse.items()}
    dataset1_var = per_tensor_mean("dataset1_variance")

    nrmse = calc_nrmse(rmse, dataset1_var)

    if per_tensor_and_channel:
        return mae, nrmse

    return dict_mean(mae), dict_mean(nrmse)
