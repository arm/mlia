# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Numpy TFRecord utils."""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Callable

import tensorflow as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def decode_fn(record_bytes: Any, type_map: dict) -> dict:
    """Decode the given bytes into a name-tensor dict assuming the given type."""
    parse_dict = {
        name: tf.io.FixedLenFeature([], tf.string) for name in type_map.keys()
    }
    example = tf.io.parse_single_example(record_bytes, parse_dict)
    features = {
        n: tf.io.parse_tensor(example[n], tf.as_dtype(t)) for n, t in type_map.items()
    }
    return features


def make_decode_fn(filename: str) -> Callable:
    """Make decode filename."""
    meta_filename = filename + ".meta"
    with open(meta_filename, encoding="utf-8") as file:
        type_map = json.load(file)["type_map"]
    return lambda record_bytes: decode_fn(record_bytes, type_map)


def numpytf_read(filename: str | Path) -> Any:
    """Read TFRecord dataset."""
    decode = make_decode_fn(str(filename))
    dataset = tf.data.TFRecordDataset(str(filename))
    return dataset.map(decode)


@lru_cache
def numpytf_count(filename: str | Path) -> int:
    """Return count from TFRecord file."""
    meta_filename = f"{filename}.meta"
    try:
        with open(meta_filename, encoding="utf-8") as file:
            return int(json.load(file)["count"])
    except FileNotFoundError:
        raw_dataset = tf.data.TFRecordDataset(filename)
        return sum(1 for _ in raw_dataset)


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
