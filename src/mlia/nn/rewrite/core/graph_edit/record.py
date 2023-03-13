# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tqdm import tqdm
from mlia.nn.rewrite.core.utils.numpy_tfrecord import (
    NumpyTFReader,
    NumpyTFWriter,
    TFLiteModel,
    numpytf_count,
)
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel


def record_model(
    input_filename,
    model_filename,
    output_filename,
    batch_size=None,
    show_progress=False,
    num_procs=1,
    num_threads=0,
):
    """num_procs: 0 => detect real cores on system
    num_threads: 0 => TFLite impl. specific setting, usually 3"""
    model = ParallelTFLiteModel(model_filename, num_procs, num_threads, batch_size)
    if not batch_size:
        batch_size = (
            model.num_procs * model.batch_size
        )  # automatically batch to the minimum effective size if not specified

    total = numpytf_count(input_filename)
    dataset = NumpyTFReader(input_filename)
    writer = NumpyTFWriter(output_filename)

    if batch_size > 1:
        # Collapse batch-size 1 items into batch-size n. I regret using batch-size 1 items in tfrecs now.
        dataset = dataset.map(
            lambda d: {k: tf.squeeze(v, axis=0) for k, v in d.items()}
        )
        dataset = dataset.batch(batch_size, drop_remainder=False)
        total = int(math.ceil(total / batch_size))

    for j, named_x in enumerate(
        tqdm(dataset.as_numpy_iterator(), total=total, disable=not show_progress)
    ):
        named_y = model(named_x)
        if batch_size > 1:
            for i in range(batch_size):
                # Expand the batches and recreate each dict as a batch-size 1 item for the tfrec output
                d = {k: v[i : i + 1] for k, v in named_y.items() if i < v.shape[0]}
                if d:
                    writer.write(d)
        else:
            writer.write(named_y)
    model.close()
