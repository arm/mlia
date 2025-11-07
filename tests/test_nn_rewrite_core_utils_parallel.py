# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for mlia.nn.rewrite.core.utils.parallel"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel
from mlia.nn.tensorflow.config import TFLiteModel

MOCKED_N_CPUS = 8


def _get_named_batch(
    data: list[dict[str, np.ndarray]], key: str
) -> dict[str, np.ndarray]:
    tensor_list = [named_input[key] for named_input in data]
    batch = np.concatenate(tensor_list, axis=0)

    return {key: batch}


@pytest.fixture(name="patch_cpu_count", autouse=True)
def patch_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock a machine with MOCKED_N_CPUS cores."""
    monkeypatch.setattr(
        "mlia.nn.rewrite.core.utils.parallel.cpu_count",
        MagicMock(return_value=MOCKED_N_CPUS),
    )


@pytest.mark.parametrize(
    "n_procs, n_threads, batch_size",
    [
        (MOCKED_N_CPUS, MOCKED_N_CPUS, 32),
        (1, MOCKED_N_CPUS, 32),
        (1, MOCKED_N_CPUS, None),
        (0, 1, None),
    ],
)
def test_parallel_tflite_model(
    test_tflite_model: Path,
    test_tfrecord: Path,
    n_procs: int,
    n_threads: int,
    batch_size: int,
) -> None:
    "Test ParallelTFLiteModel class."
    model = ParallelTFLiteModel(test_tflite_model, n_procs, n_threads, batch_size)
    assert model.num_procs == n_procs if n_procs > 0 else MOCKED_N_CPUS
    assert model.num_threads == n_threads
    if model.num_procs > 1:  # Parallel execution
        assert (
            model.batch_size == int(math.ceil(batch_size / model.num_procs))
            if batch_size
            else 1
        )
    else:  # Fall back to serial execution
        assert (
            model.batch_size
            == TFLiteModel(test_tflite_model, batch_size, n_threads).batch_size
        )

    input_data = list(numpytf_read(test_tfrecord).as_numpy_iterator())

    assert model(input_data[0])["StatefulPartitionedCall:0"].shape[0] == 1
    assert model(_get_named_batch(input_data, "serving_default_input:0"))[
        "StatefulPartitionedCall:0"
    ].shape[0] == len(input_data)


def test_process_utilization_warning(
    test_tflite_model: Path,
    test_tfrecord: Path,
) -> None:
    """Test if warning is shown if parallelism is not fully utilized."""
    model = ParallelTFLiteModel(test_tflite_model, 8, 8)
    input_data = list(numpytf_read(test_tfrecord).as_numpy_iterator())
    input_batch = _get_named_batch(input_data, "serving_default_input:0")

    for _ in range(11):
        _ = model(input_batch)

    assert model.warned
