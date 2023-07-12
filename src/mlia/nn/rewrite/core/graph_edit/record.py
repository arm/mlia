# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Save subgraph data."""
# pylint: disable=too-many-locals
from __future__ import annotations

import math
import os
from contextlib import ExitStack
from pathlib import Path

import tensorflow as tf
from rich.progress import track

from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_count
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read
from mlia.nn.rewrite.core.utils.numpy_tfrecord import NumpyTFWriter
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel
from mlia.nn.tensorflow.config import NameToTensorMap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DEQUANT_SUFFIX = "_dequant"


def dequantized_path(filename: str | Path) -> Path:
    """Append the de-quantization suffix to the given filename."""
    path = Path(filename)
    path = Path(path.parent, f"{path.stem}{DEQUANT_SUFFIX}{path.suffix}")
    return path


def record_model(
    input_filename: str | Path,
    model_filename: str | Path,
    output_filename: str | Path,
    batch_size: int = 0,
    show_progress: bool = False,
    num_procs: int = 1,
    num_threads: int = 0,
    dequantize_output: bool = False,
) -> None:
    """Model recorder.

    num_procs: 0 => detect real cores on system
    num_threads: 0 => TFLite impl. specific setting, usually 3

    dequantize: True => de-quantize the recorded output before saving
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

    with ExitStack() as stack:
        writer = stack.enter_context(NumpyTFWriter(output_filename))
        writer_dequant = None
        if dequantize_output:
            dequant_path = dequantized_path(output_filename)
            writer_dequant = stack.enter_context(NumpyTFWriter(dequant_path))

        def write(writer: NumpyTFWriter, data: NameToTensorMap) -> None:
            """Write the data using the given NumpyTFWriter instance."""
            if batch_size > 1:
                for i in range(batch_size):
                    # Expand the batches and recreate each dict as a
                    # batch-size 1 item for the tfrec output
                    recreated_dict = {
                        k: v[i : i + 1]  # noqa: E203
                        for k, v in data.items()
                        if i < v.shape[0]
                    }
                    if recreated_dict:
                        writer.write(recreated_dict)
            else:
                writer.write(data)

        for _, named_x in enumerate(
            track(dataset.as_numpy_iterator(), total=total, disable=not show_progress)
        ):
            named_y = model(named_x)
            write(writer, named_y)

            if dequantize_output:
                assert writer_dequant
                named_y_dequant = model.dequantize_outputs(named_y)
                write(writer_dequant, named_y_dequant)

        model.close()
