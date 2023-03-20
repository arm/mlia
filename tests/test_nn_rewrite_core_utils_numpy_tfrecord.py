# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.utils.numpy_tfrecord."""
from __future__ import annotations

from pathlib import Path

from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_count
from mlia.nn.rewrite.core.utils.numpy_tfrecord import sample_tfrec


def test_sample_tfrec(test_tfrecord: Path, tmp_path: Path) -> None:
    """Test function sample_tfrec()."""
    output_file = tmp_path / "output.tfrecord"
    # Sample 1 sample from test_tfrecord
    sample_tfrec(input_file=str(test_tfrecord), k=1, output_file=str(output_file))
    assert output_file.is_file()
    assert numpytf_count(str(output_file)) == 1
