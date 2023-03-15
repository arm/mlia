# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.graph_edit.cut."""
from pathlib import Path

import numpy as np
import tensorflow as tf

from mlia.nn.rewrite.core.graph_edit.cut import cut_model


def test_cut_model(test_tflite_model: Path, tmp_path: Path) -> None:
    """Test the function cut_model()."""
    output_file = tmp_path / "out.tflite"
    cut_model(
        model_file=str(test_tflite_model),
        input_names=["serving_default_input:0"],
        output_names=["sequential/flatten/Reshape"],
        subgraph_index=0,
        output_file=str(output_file),
    )
    assert output_file.is_file()

    interpreter = tf.lite.Interpreter(model_path=str(output_file))
    output_details = interpreter.get_output_details()
    assert len(output_details) == 1
    out = output_details[0]
    assert "Reshape" in out["name"]
    assert np.prod(out["shape"]) == 1728
