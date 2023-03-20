# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from tensorflow.lite.python import interpreter as interpreter_wrapper
from mlia.nn.rewrite.core.utils.numpy_tfrecord import NumpyTFReader, NumpyTFWriter


def diff_stats(file1, file2, per_tensor_and_channel=False):
    dataset1 = NumpyTFReader(file1)
    dataset2 = NumpyTFReader(file2)

    totals = defaultdict(dict)

    def add_total(name, key, values):
        if not key in totals[name]:
            totals[name][key] = values
        else:
            totals[name][key] += values

    # First iterate through dataset1 and calculate per-channel total for each tensor
    count = 0
    for d in dataset1:
        count += 1
        for k, v in d.items():
            value = v.numpy().astype(np.double)
            add_total("dataset1_total", k, value)

    # Use this to calculate per-channel mean for each tensor
    per_tensor_mean = lambda name: {
        k: total / count for k, total in totals[name].items()
    }
    dataset1_mean = per_tensor_mean("dataset1_total")

    # Next iterate through both datasets and calculate per-channel total squared error
    # between them for each tensor and dataset1 variance for each tensor using the mean from above
    for i, (x1, x2) in enumerate(zip(dataset1, dataset2)):
        assert x1.keys() == x2.keys(), (
            "At input %d the files have different sets of tensors.\n%s: %s\n%s: %s\n"
            % (
                i,
                file1,
                ", ".join(x1.keys()),
                file2,
                ", ".join(x2.keys()),
            )
        )
        for k in x1.keys():
            v1 = x1[k].numpy().astype(np.double)
            v2 = x2[k].numpy().astype(np.double)
            add_total("ae", k, abs(v1 - v2))
            add_total("se", k, (v1 - v2) ** 2)
            add_total("dataset1_variance", k, (v1 - dataset1_mean[k]) ** 2)

    # Finally average over number of inputs to get the rmse and the dataset1 variance
    mae = per_tensor_mean("ae")
    mse = per_tensor_mean("se")
    rmse = {k: np.sqrt(v) for k, v in mse.items()}
    dataset1_var = per_tensor_mean("dataset1_variance")
    is_nonzero = {k: dataset1_var[k] > 0 for k in dataset1_var}

    # Divide by target standard deviation to get the per-channel nrmse for each tensor where possible
    nrmse = {
        k: v[is_nonzero[k]] / np.sqrt(dataset1_var[k][is_nonzero[k]])
        for k, v in rmse.items()
    }

    if per_tensor_and_channel:
        return mae, nrmse
    else:
        dict_mean = lambda d: np.mean(list(d.values()))
        return dict_mean(mae), dict_mean(nrmse)
