# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common test utils for the rewrite tests."""
from __future__ import annotations

from tensorflow.lite.python.schema_py_generated import ModelT


def models_are_equal(model1: ModelT, model2: ModelT) -> bool:
    """Check that the two models are equal."""
    if len(model1.subgraphs) != len(model2.subgraphs):
        return False

    for graph1, graph2 in zip(model1.subgraphs, model2.subgraphs):
        if graph1.name != graph2.name or len(graph1.tensors) != len(graph2.tensors):
            return False
        for tensor1 in graph1.tensors:
            for tensor2 in graph2.tensors:
                if tensor1.name == tensor2.name:
                    if (
                        tensor1.shape == tensor2.shape
                    ).all() or tensor1.type == tensor2.type:
                        break
            else:
                return False  # Tensor from graph1 not found in other graph.")

    return True
