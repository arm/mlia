# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.utils.numpy_tfrecord."""
from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

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
