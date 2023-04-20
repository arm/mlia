# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.graph_edit.join."""
from pathlib import Path

import numpy as np
import pytest

from mlia.nn.rewrite.core.graph_edit.diff import calc_nrmse
from mlia.nn.rewrite.core.graph_edit.diff import diff_stats


def assert_two_dictionaries_with_numpy_arrays(dict1: dict, dict2: dict) -> None:
    """Use numpy assertions to check whether two dictionaries are equal."""
    key1, val1 = list(dict1.keys()), list(dict1.values())
    key2, val2 = list(dict2.keys()), list(dict2.values())
    np.testing.assert_equal(key1, key2)
    np.testing.assert_almost_equal(val1, val2)


@pytest.mark.parametrize(
    "rmse, scale, expected_result",
    [
        (
            {"test1": np.ndarray((3,), buffer=np.array([1.0, 2.0, 3.3]))},
            {"test1": np.ndarray((3,), buffer=np.array([4.0, 4.0, 0.0]))},
            {"test1": np.ndarray((3,), buffer=np.array([0.5, 1.0, 3.3]))},
        ),
        (
            {"test1": np.ndarray((3,), buffer=np.array([1.0, 2.0, 3.3]))},
            {"test1": np.ndarray((3,), buffer=np.array([0.0, 0.0, 0.0]))},
            {"test1": np.ndarray((3,), buffer=np.array([1.0, 2.0, 3.3]))},
        ),
    ],
)
def test_calc_nrmse(rmse: dict, scale: dict, expected_result: dict) -> None:
    """Test calc_nrmse() function."""
    assert_two_dictionaries_with_numpy_arrays(calc_nrmse(rmse, scale), expected_result)


def test_diff_stats(test_tfrecord_fp32: Path) -> None:
    """Test diff_stats() function."""
    res1, res2 = diff_stats(test_tfrecord_fp32, test_tfrecord_fp32)
    assert res1 == 0.0
    assert res2 == 0.0
