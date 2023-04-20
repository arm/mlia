# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.rewrite."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest

from mlia.nn.rewrite.core.rewrite import RewriteConfiguration
from mlia.nn.rewrite.core.rewrite import Rewriter
from mlia.nn.tensorflow.config import TFLiteModel


@pytest.mark.parametrize(
    "rewrite_name, expected_error",
    [
        ("fully_connected", does_not_raise()),
        ("random", does_not_raise()),
    ],
)
def test_rewrite_configuration(
    test_tflite_model_fp32: Path, rewrite_name: str, expected_error: Any
) -> None:
    """Test get_rewrite function only supports rewrite type fully_connected."""
    with expected_error:
        config_obj = RewriteConfiguration(
            rewrite_name,
            ["sample_node_start", "sample_node_end"],
            None,
        )

        rewriter_obj = Rewriter(test_tflite_model_fp32, config_obj)
        assert rewriter_obj.optimizer_configuration.optimization_target == rewrite_name
        assert isinstance(rewriter_obj, Rewriter)


def test_rewriter(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
) -> None:
    """Test fc_layer rewrite process with rewrite type fully_connected."""
    config_obj = RewriteConfiguration(
        "fully_connected",
        ["sequential/flatten/Reshape", "StatefulPartitionedCall:0"],
        test_tfrecord_fp32,
    )

    test_obj = Rewriter(test_tflite_model_fp32, config_obj)
    test_obj.apply_optimization()
    trained_model = test_obj.get_model()

    assert isinstance(trained_model, TFLiteModel)
