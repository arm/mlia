# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Extract module."""
# pylint: disable=too-many-arguments, too-many-locals
import os

import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import SubGraphT

from mlia.nn.rewrite.core.graph_edit.cut import cut_model
from mlia.nn.rewrite.core.graph_edit.record import record_model


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def extract(
    output_path: str,
    model_file: str,
    input_filename: str,
    input_names: list,
    output_names: list,
    subgraph: SubGraphT = 0,
    skip_outputs: bool = False,
    show_progress: bool = False,
    num_procs: int = 1,
    num_threads: int = 0,
) -> None:
    """Extract a model after cut and record."""
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    start_file = os.path.join(output_path, "start.tflite")
    cut_model(
        model_file,
        input_names=None,
        output_names=input_names,
        subgraph_index=subgraph,
        output_file=start_file,
    )

    input_tfrec = os.path.join(output_path, "input.tfrec")
    record_model(
        input_filename,
        start_file,
        input_tfrec,
        show_progress=show_progress,
        num_procs=num_procs,
        num_threads=num_threads,
    )

    replace_file = os.path.join(output_path, "replace.tflite")
    cut_model(
        model_file,
        input_names=input_names,
        output_names=output_names,
        subgraph_index=subgraph,
        output_file=replace_file,
    )

    end_file = os.path.join(output_path, "end.tflite")
    cut_model(
        model_file,
        input_names=output_names,
        output_names=None,
        subgraph_index=subgraph,
        output_file=end_file,
    )

    if not skip_outputs:
        output_tfrec = os.path.join(output_path, "output.tfrec")
        record_model(
            input_tfrec,
            replace_file,
            output_tfrec,
            show_progress=show_progress,
            num_procs=num_procs,
            num_threads=num_threads,
        )

        end_tfrec = os.path.join(output_path, "end.tfrec")
        record_model(
            output_tfrec,
            end_file,
            end_tfrec,
            show_progress=show_progress,
            num_procs=num_procs,
            num_threads=num_threads,
        )
