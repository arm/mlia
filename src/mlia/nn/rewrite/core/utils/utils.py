# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import os

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb


def load(input_tflite_file):
    if not os.path.exists(input_tflite_file):
        raise RuntimeError("TFLite file not found at %r\n" % input_tflite_file)
    with open(input_tflite_file, "rb") as file_handle:
        file_data = bytearray(file_handle.read())
    model_obj = schema_fb.Model.GetRootAsModel(file_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    return model


def save(model, output_tflite_file):
    builder = flatbuffers.Builder(1024)  # Initial size of the buffer, which
    # will grow automatically if needed
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    model_data = builder.Output()
    with open(output_tflite_file, "wb") as out_file:
        out_file.write(model_data)
