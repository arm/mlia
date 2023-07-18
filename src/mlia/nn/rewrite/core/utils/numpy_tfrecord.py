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

from mlia.core.errors import ConfigurationError
from mlia.nn.tensorflow.utils import get_tflite_model_type_map

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


def make_decode_fn(filename: str, model_filename: str | Path | None = None) -> Callable:
    """Make decode filename."""
    meta_filename = filename + ".meta"
    try:
        with open(meta_filename, encoding="utf-8") as file:
            type_map = json.load(file)["type_map"]
    except FileNotFoundError:
        if model_filename is None:
            raise
        type_map = get_tflite_type_map(str(model_filename), filename)
    return lambda record_bytes: decode_fn(record_bytes, type_map)


def get_tflite_type_map(
    model_filename: str | Path, tfrec_filename: str | Path
) -> dict[str, type]:
    """Get type map from input model."""
    model_type_map = get_tflite_model_type_map(model_filename)
    get_tfrecord_type_map(tfrec_filename, model_type_map)
    return model_type_map


def get_tfrecord_type_map(filename: str | Path, model_type_map: dict) -> dict[str, str]:
    """Crosscheck type map with tfrecord file."""

    def parse_tensor_and_update_type_map(example_proto: Any, model_dtype: Any) -> bool:
        """Attempt to parse tensor with given tf dtype."""
        try:
            tf.io.parse_tensor(example_proto, model_dtype)
            tfrec_type_map[key] = model_dtype
            return True
        except tf.errors.InvalidArgumentError as error_output:
            check_type_match(model_dtype, error_output)
        return False

    dataset = tf.data.TFRecordDataset(str(filename))
    example = tf.train.Example()
    tfrec_type_map: dict = {}
    model_dtype = list(model_type_map.values())[0]

    for record in dataset:
        example.ParseFromString(record.numpy())
        feature_map = example.features.feature  # pylint: disable=E1101
        for key, example_proto in feature_map.items():
            if example_proto.HasField("bytes_list"):
                parse_tensor_and_update_type_map(
                    example_proto.bytes_list.value[0],
                    model_dtype,
                )
            else:
                error_output = "Unknown tfrecord file format."
                if example_proto.HasField("float_list"):
                    tfrec_type_map[key] = tf.float32
                    error_output = (
                        f"Model type {model_dtype} does not match "
                        f"{tfrec_type_map[key]}."
                    )
                elif example_proto.HasField("int64_list"):
                    tfrec_type_map[key] = tf.int64
                    error_output = (
                        f"Model type {model_dtype} does "
                        f"not match {tfrec_type_map[key]}."
                    )
                check_type_match(model_dtype, error_output, tfrec_type_map[key])
            break
    return tfrec_type_map


def check_type_match(
    model_dtype: Any, error_output: str, tfrec_dtype: Any = None
) -> bool:
    """Check type match and raise error."""
    if tfrec_dtype != model_dtype:
        error_message = (
            "The provided dataset does not match the TensorFLow Lite model: "
            + str(model_dtype)
            + "\nEither use a different matching dataset or "
            + "re-write your dataset accordingly using"
            + " NumpyTFWriter from mlia.nn.rewrite.core.utils.numpytfrecord.\n\n"
            + str(error_output)
        )
        raise ConfigurationError(error_message)
    return True


def numpytf_read(filename: str | Path, model_filename: str | Path | None = None) -> Any:
    """Read TFRecord dataset."""
    decode = make_decode_fn(str(filename), model_filename)
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
