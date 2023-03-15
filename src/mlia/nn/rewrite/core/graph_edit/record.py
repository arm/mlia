# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Save subgraph data."""
# pylint: disable=too-many-locals
from __future__ import annotations

import math
import os
from pathlib import Path

import tensorflow as tf
from rich.progress import track

from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_count
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read
from mlia.nn.rewrite.core.utils.numpy_tfrecord import NumpyTFWriter
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def record_model(
    input_filename: str | Path,
    model_filename: str | Path,
    output_filename: str | Path,
    batch_size: int = 0,
    show_progress: bool = False,
    num_procs: int = 1,
    num_threads: int = 0,
) -> None:
    """Model recorder.

    num_procs: 0 => detect real cores on system
    num_threads: 0 => TFLite impl. specific setting, usually 3
    """
    model = ParallelTFLiteModel(model_filename, num_procs, num_threads, batch_size)
    if not batch_size:
        batch_size = (
            model.num_procs * model.batch_size
        )  # automatically batch to the minimum effective size if not specified

    total = numpytf_count(input_filename)
    dataset = numpytf_read(input_filename)

    if batch_size > 1:
        # Collapse batch-size 1 items into batch-size n.
        dataset = dataset.map(
            lambda d: {k: tf.squeeze(v, axis=0) for k, v in d.items()}
        )
        dataset = dataset.batch(batch_size, drop_remainder=False)
        total = int(math.ceil(total / batch_size))

    with NumpyTFWriter(output_filename) as writer:
        for _, named_x in enumerate(
            track(dataset.as_numpy_iterator(), total=total, disable=not show_progress)
        ):
            named_y = model(named_x)
            if batch_size > 1:
                for i in range(batch_size):
                    # Expand the batches and recreate each dict as a
                    # batch-size 1 item for the tfrec output
                    recreated_dict = {
                        k: v[i : i + 1]  # noqa: E203
                        for k, v in named_y.items()
                        if i < v.shape[0]
                    }
                    if recreated_dict:
                        writer.write(recreated_dict)
            else:
                writer.write(named_y)
        model.close()
