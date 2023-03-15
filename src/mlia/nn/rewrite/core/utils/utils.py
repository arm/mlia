# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model and file system utilites."""
from __future__ import annotations

from pathlib import Path

import flatbuffers
from tensorflow.lite.python.schema_py_generated import Model
from tensorflow.lite.python.schema_py_generated import ModelT


def load(input_tflite_file: str | Path) -> ModelT:
    """Load a flatbuffer model from file."""
    if not Path(input_tflite_file).exists():
        raise FileNotFoundError(f"TFLite file not found at {input_tflite_file}\n")
    with open(input_tflite_file, "rb") as file_handle:
        file_data = bytearray(file_handle.read())
    model_obj = Model.GetRootAsModel(file_data, 0)
    model = ModelT.InitFromObj(model_obj)
    return model


def save(model: ModelT, output_tflite_file: str | Path) -> None:
    """Save a flatbuffer model to a given file."""
    builder = flatbuffers.Builder(1024)  # Initial size of the buffer, which
    # will grow automatically if needed
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    model_data = builder.Output()
    with open(output_tflite_file, "wb") as out_file:
        out_file.write(model_data)
