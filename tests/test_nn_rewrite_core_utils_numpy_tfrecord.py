# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.utils.numpy_tfrecord."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mlia.core.errors import ConfigurationError
from mlia.nn.rewrite.core.utils.numpy_tfrecord import check_type_match
from mlia.nn.rewrite.core.utils.numpy_tfrecord import get_tflite_type_map
from mlia.nn.rewrite.core.utils.numpy_tfrecord import make_decode_fn
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_count
from mlia.nn.rewrite.core.utils.numpy_tfrecord import sample_tfrec


def test_sample_tfrec(test_tfrecord: Path, tmp_path: Path) -> None:
    """Test function sample_tfrec()."""
    output_file = tmp_path / "output.tfrecord"
    # Sample 1 sample from test_tfrecord
    sample_tfrec(input_file=str(test_tfrecord), k=1, output_file=str(output_file))
    assert output_file.is_file()
    assert numpytf_count(str(output_file)) == 1


def test_get_tflite_type_map(
    test_tflite_model: Path,
    test_tfrecord: Path,
) -> None:
    """Test function get_type_map()."""
    type_map = get_tflite_type_map(test_tflite_model, test_tfrecord)
    for key, _ in type_map.items():
        assert type_map[key] is np.int8
        assert type_map


def test_get_tflite_type_map_float_list(
    test_tflite_model_fp32: Path,
    test_tfrecord_float_list: Path,
) -> None:
    """Test function get_type_map() with a float list tfrecord."""
    type_map = get_tflite_type_map(test_tflite_model_fp32, test_tfrecord_float_list)
    for key, _ in type_map.items():
        assert type_map[key] is np.float32
        assert type_map


def test_check_type_match() -> None:
    """Test function check_type_map()."""
    sample_error = "Sample error output"
    assert check_type_match(tf.float32, sample_error, tf.float32) is True
    with pytest.raises(ConfigurationError):
        check_type_match(tf.int8, sample_error, tf.float32)


def test_make_decode_fn(test_tfrecord: Path) -> None:
    """Test function make_decode_fn()."""
    decode = make_decode_fn(str(test_tfrecord))
    dataset = tf.data.TFRecordDataset(str(test_tfrecord))
    features = decode(next(iter(dataset)))
    assert isinstance(features, dict)
    assert len(features) == 1
    key, val = next(iter(features.items()))
    assert isinstance(key, str)
    assert isinstance(val, tf.Tensor)
    assert val.dtype == tf.int8

    with pytest.raises(FileNotFoundError):
        make_decode_fn(str(test_tfrecord) + "_")


def test_numpytf_count(test_tfrecord: Path) -> None:
    """Test function numpytf_count()."""
    assert numpytf_count(test_tfrecord) == 3
