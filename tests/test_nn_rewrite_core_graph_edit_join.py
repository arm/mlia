# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.graph_edit.join."""
from pathlib import Path

from mlia.nn.rewrite.core.graph_edit.cut import cut_model
from mlia.nn.rewrite.core.graph_edit.join import join_models
from mlia.nn.rewrite.core.utils.utils import load
from tests.utils.rewrite import models_are_equal


def test_join_model(test_tflite_model: Path, tmp_path: Path) -> None:
    """Test the function join_models()."""
    # Cut model into two parts first
    first_file = tmp_path / "first_part.tflite"
    second_file = tmp_path / "second_part.tflite"
    cut_model(
        model_file=str(test_tflite_model),
        input_names=["serving_default_input:0"],
        output_names=["sequential/flatten/Reshape"],
        subgraph_index=0,
        output_file=str(first_file),
    )
    cut_model(
        model_file=str(test_tflite_model),
        input_names=["sequential/flatten/Reshape"],
        output_names=["StatefulPartitionedCall:0"],
        subgraph_index=0,
        output_file=str(second_file),
    )
    assert first_file.is_file()
    assert second_file.is_file()

    joined_file = tmp_path / "joined.tflite"

    # Now re-join the cut model and check the result is the same as the original
    for in_src, in_dst in ((first_file, second_file), (second_file, first_file)):
        join_models(
            input_src=str(in_src),
            input_dst=str(in_dst),
            output_file=str(joined_file),
            subgraph_src=0,
            subgraph_dst=0,
        )
        assert joined_file.is_file()

        orig_model = load(str(test_tflite_model))
        joined_model = load(str(joined_file))

        assert models_are_equal(orig_model, joined_model)
