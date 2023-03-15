# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Numpy TFRecord utils."""
from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper

from mlia.nn.rewrite.core.utils.utils import load
from mlia.nn.rewrite.core.utils.utils import save

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def make_decode_fn(filename: str) -> Callable:
    """Make decode filename."""

    def decode_fn(record_bytes: Any, type_map: dict) -> dict:
        parse_dict = {
            name: tf.io.FixedLenFeature([], tf.string) for name in type_map.keys()
        }
        example = tf.io.parse_single_example(record_bytes, parse_dict)
        features = {
            n: tf.io.parse_tensor(example[n], tf.as_dtype(t))
            for n, t in type_map.items()
        }
        return features

    meta_filename = filename + ".meta"
    with open(meta_filename, encoding="utf-8") as file:
        type_map = json.load(file)["type_map"]
    return lambda record_bytes: decode_fn(record_bytes, type_map)


def numpytf_read(filename: str | Path) -> Any:
    """Read TFRecord dataset."""
    decode_fn = make_decode_fn(str(filename))
    dataset = tf.data.TFRecordDataset(str(filename))
    return dataset.map(decode_fn)


def numpytf_count(filename: str | Path) -> Any:
    """Return count from TFRecord file."""
    meta_filename = f"{filename}.meta"
    with open(meta_filename, encoding="utf-8") as file:
        return json.load(file)["count"]


class NumpyTFWriter:
    """Numpy TF serializer."""

    def __init__(self, filename: str | Path) -> None:
        """Initiate a Numpy TF Serializer."""
        self.filename = filename
        self.meta_filename = f"{filename}.meta"
        self.writer = tf.io.TFRecordWriter(str(filename))
        self.type_map: dict = {}
        self.count = 0

    def __enter__(self) -> Any:
        """Enter instance."""
        return self

    def __exit__(
        self, exception_type: Any, exception_value: Any, exception_traceback: Any
    ) -> None:
        """Close instance."""
        self.close()

    def write(self, array_dict: dict) -> None:
        """Write array dict."""
        type_map = {n: str(a.dtype.name) for n, a in array_dict.items()}
        self.type_map.update(type_map)
        self.count += 1

        feature = {
            n: tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(a).numpy()])
            )
            for n, a in array_dict.items()
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

    def close(self) -> None:
        """Close NumpyTFWriter."""
        with open(self.meta_filename, "w", encoding="utf-8") as file:
            meta = {"type_map": self.type_map, "count": self.count}
            json.dump(meta, file)
        self.writer.close()


class TFLiteModel:
    """A representation of a TFLite Model."""

    def __init__(
        self,
        filename: str,
        batch_size: int | None = None,
        num_threads: int | None = None,
    ) -> None:
        """Initiate a TFLite Model."""
        if not num_threads:
            num_threads = None
        if not batch_size:
            self.interpreter = interpreter_wrapper.Interpreter(
                model_path=filename, num_threads=num_threads
            )
        else:  # if a batch size is specified, modify the TFLite model to use this size
            with tempfile.TemporaryDirectory() as tmp:
                flatbuffer = load(filename)
                for subgraph in flatbuffer.subgraphs:
                    for tensor in list(subgraph.inputs) + list(subgraph.outputs):
                        subgraph.tensors[tensor].shape = np.array(
                            [batch_size] + list(subgraph.tensors[tensor].shape[1:]),
                            dtype=np.int32,
                        )
                tempname = os.path.join(tmp, "rewrite_tmp.tflite")
                save(flatbuffer, tempname)
                self.interpreter = interpreter_wrapper.Interpreter(
                    model_path=tempname, num_threads=num_threads
                )

        try:
            self.interpreter.allocate_tensors()
        except RuntimeError:
            self.interpreter = interpreter_wrapper.Interpreter(
                model_path=filename, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        details = list(self.input_details) + list(self.output_details)
        self.handle_from_name = {d["name"]: d["index"] for d in details}
        self.shape_from_name = {d["name"]: d["shape"] for d in details}
        self.batch_size = next(iter(self.shape_from_name.values()))[0]

    def __call__(self, named_input: dict) -> dict:
        """Execute the model on one or a batch of named inputs \
            (a dict of name: numpy array)."""
        input_len = next(iter(named_input.values())).shape[0]
        full_steps = input_len // self.batch_size
        remainder = input_len % self.batch_size

        named_ys = defaultdict(list)
        for i in range(full_steps):
            for name, x_batch in named_input.items():
                x_tensor = x_batch[i : i + self.batch_size]  # noqa: E203
                self.interpreter.set_tensor(self.handle_from_name[name], x_tensor)
            self.interpreter.invoke()
            for output_detail in self.output_details:
                named_ys[output_detail["name"]].append(
                    self.interpreter.get_tensor(output_detail["index"])
                )
        if remainder:
            for name, x_batch in named_input.items():
                x_tensor = np.zeros(  # pylint: disable=invalid-name
                    self.shape_from_name[name]
                ).astype(x_batch.dtype)
                x_tensor[:remainder] = x_batch[-remainder:]
                self.interpreter.set_tensor(self.handle_from_name[name], x_tensor)
            self.interpreter.invoke()
            for output_detail in self.output_details:
                named_ys[output_detail["name"]].append(
                    self.interpreter.get_tensor(output_detail["index"])[:remainder]
                )
        return {k: np.concatenate(v) for k, v in named_ys.items()}

    def input_tensors(self) -> list:
        """Return name from input details."""
        return [d["name"] for d in self.input_details]

    def output_tensors(self) -> list:
        """Return name from output details."""
        return [d["name"] for d in self.output_details]


def sample_tfrec(input_file: str, k: int, output_file: str) -> None:
    """Count, read and write TFRecord input and output data."""
    total = numpytf_count(input_file)
    next_sample = sorted(random.sample(range(total), k=k), reverse=True)

    reader = numpytf_read(input_file)
    with NumpyTFWriter(output_file) as writer:
        for i, data in enumerate(reader):
            if i == next_sample[-1]:
                next_sample.pop()
                writer.write(data)
                if not next_sample:
                    break
