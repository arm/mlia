# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Extract module."""
# pylint: disable=too-many-arguments, too-many-locals
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import SubGraphT

from mlia.nn.rewrite.core.graph_edit.cut import cut_model
from mlia.nn.rewrite.core.graph_edit.record import dequantized_path
from mlia.nn.rewrite.core.graph_edit.record import record_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _get_path(
    ext: str, name: str, dir_path: str | Path, model_is_quantized: bool = False
) -> Path:
    """Create a file path for extracted files."""
    path = Path(dir_path, f"{name}{ext}")
    return dequantized_path(path) if model_is_quantized else path


class TFLitePaths:  # pylint: disable=too-few-public-methods
    """Provide safe access to TensorFlow Lite file paths."""

    _get_path_tflite = partial(_get_path, ".tflite")

    start = partial(_get_path_tflite, "start")
    replace = partial(_get_path_tflite, "replace")
    end = partial(_get_path_tflite, "end")


class TFRecordPaths:  # pylint: disable=too-few-public-methods
    """Provide safe access to tfrec file paths."""

    _get_path_tfrec = partial(_get_path, ".tfrec")

    input = partial(_get_path_tfrec, "input")
    output = partial(_get_path_tfrec, "output")
    end = partial(_get_path_tfrec, "end")


class ExtractPaths:  # pylint: disable=too-few-public-methods
    """Get paths to extract files.

    This is meant to be the single source of truth regarding all file names
    created by the extract() function in an output directory.
    """

    tflite = TFLitePaths
    tfrec = TFRecordPaths


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
    dequantize_output: bool = False,
) -> None:
    """Extract a model after cut and record."""
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    start_file = ExtractPaths.tflite.start(output_path)
    cut_model(
        model_file,
        input_names=None,
        output_names=input_names,
        subgraph_index=subgraph,
        output_file=start_file,
    )

    input_tfrec = ExtractPaths.tfrec.input(output_path)
    record_model(
        input_filename,
        start_file,
        input_tfrec,
        show_progress=show_progress,
        num_procs=num_procs,
        num_threads=num_threads,
        dequantize_output=dequantize_output,
    )

    replace_file = ExtractPaths.tflite.replace(output_path)
    cut_model(
        model_file,
        input_names=input_names,
        output_names=output_names,
        subgraph_index=subgraph,
        output_file=replace_file,
    )

    end_file = ExtractPaths.tflite.end(output_path)
    cut_model(
        model_file,
        input_names=output_names,
        output_names=None,
        subgraph_index=subgraph,
        output_file=end_file,
    )

    if not skip_outputs:
        output_tfrec = ExtractPaths.tfrec.output(output_path)
        record_model(
            input_tfrec,
            replace_file,
            output_tfrec,
            show_progress=show_progress,
            num_procs=num_procs,
            num_threads=num_threads,
            dequantize_output=dequantize_output,
        )

        end_tfrec = ExtractPaths.tfrec.end(output_path)
        record_model(
            output_tfrec,
            end_file,
            end_tfrec,
            show_progress=show_progress,
            num_procs=num_procs,
            num_threads=num_threads,
            dequantize_output=dequantize_output,
        )
