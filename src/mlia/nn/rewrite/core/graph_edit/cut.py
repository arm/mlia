# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cut module."""
import os
from collections import defaultdict
from typing import Optional

import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import ModelT
from tensorflow.lite.python.schema_py_generated import SubGraphT

from mlia.nn.rewrite.core.utils.utils import load
from mlia.nn.rewrite.core.utils.utils import save

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def tensors_by_name(subgraph: SubGraphT, names: list) -> list:
    """Seek out tensors from a subgraph and return the result."""
    seek = frozenset([name.encode("utf-8") for name in names])
    tensors = [i for i, tensor in enumerate(subgraph.tensors) if tensor.name in seek]
    return tensors


def cut_subgraph(
    subgraph: SubGraphT,
    input_tensor_names: Optional[list],
    output_tensor_names: Optional[list],
) -> None:
    """Change the global inputs and outputs of a graph to the provided named tensors."""
    if input_tensor_names is not None:
        subgraph.inputs = tensors_by_name(subgraph, input_tensor_names)
        assert len(subgraph.inputs) == len(
            input_tensor_names
        ), f"Expected {len(subgraph.inputs)} input tensors: \
            {', '.join(input_tensor_names)}\nFound: \
            {', '.join(subgraph.tensors[i].name for i in subgraph.inputs)}"
    if output_tensor_names is not None:
        subgraph.outputs = tensors_by_name(subgraph, output_tensor_names)
        assert len(subgraph.outputs) == len(
            output_tensor_names
        ), f"Expected {len(subgraph.outputs)} output tensors: \
            {', '.join(output_tensor_names)}\nFound: \
            {', '.join(subgraph.tensors[i].name for i in subgraph.outputs)}"


def simplify(model: ModelT) -> None:
    """Remove any unused operators, tensors and buffers from a model."""
    for subgraph in model.subgraphs:
        simplify_subgraph(subgraph)

    used_buffers = {
        tensor.buffer for tensor in subgraph.tensors for subgraph in model.subgraphs
    }
    used_buffers = used_buffers.union({metadata.buffer for metadata in model.metadata})
    used_buffers.add(
        0
    )  # Buffer zero is always expected to be a zero-sized nullptr buffer by the
    # TFLite runtime
    model.buffers, buf_relabel = filter_relabel(model.buffers, used_buffers)

    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            tensor.buffer = buf_relabel[tensor.buffer]

    for metadata in model.metadata:
        metadata.buffer = buf_relabel[metadata.buffer]


def simplify_subgraph(subgraph: SubGraphT) -> None:
    """Simplify a subgraph given its operators."""
    requires = defaultdict(set)

    for output, operator in enumerate(subgraph.operators):
        for tensor in operator.outputs:
            if tensor not in subgraph.inputs:
                requires[tensor].add(output)

    op_set, ten_set = find_required(subgraph, requires, subgraph.outputs)

    subgraph.operators, _ = filter_relabel(subgraph.operators, op_set)
    subgraph.tensors, ten_relabel = filter_relabel(subgraph.tensors, ten_set)

    ten_relabel[-1] = -1  # Some files have ops with -1 input tensors; leave unchanged

    for operator in subgraph.operators:
        operator.inputs = [ten_relabel[tensor] for tensor in operator.inputs]
        operator.outputs = [ten_relabel[tensor] for tensor in operator.outputs]

    subgraph.inputs = [ten_relabel[tensor] for tensor in subgraph.inputs]
    subgraph.outputs = [ten_relabel[tensors] for tensors in subgraph.outputs]


def find_required(subgraph: SubGraphT, requires: dict, tensors: dict) -> tuple:
    """Find required operators in a given subgraph."""
    visited_operators: set = set()
    visited_tensors = set(tensors)
    stop_tensors = set(subgraph.inputs)

    next_tensors = visited_tensors
    while next_tensors:
        loop_tensors = next_tensors
        next_tensors = set()
        for tensor in loop_tensors:
            candidate_operators = set(requires[tensor])
            new_operators = candidate_operators - visited_operators
            visited_operators = visited_operators.union(new_operators)
            for operator in new_operators:
                candidate_tensors = set(subgraph.operators[operator].inputs)
                new_tensors = candidate_tensors - (visited_tensors.union(next_tensors))
                next_tensors = next_tensors.union(new_tensors)
                visited_tensors = visited_tensors.union(candidate_tensors)
                visited_tensors = visited_tensors.union(
                    subgraph.operators[operator].outputs
                )  # include stub outputs but do not traverse them
        next_tensors = next_tensors - stop_tensors

    return visited_operators, visited_tensors


def filter_relabel(src_subgraph: SubGraphT, relabel_filter: set) -> tuple:
    """Relabel tensors in a subgraph based on a filter."""
    relabel: dict = {}
    output: list = []
    for i, out in enumerate(src_subgraph):
        if i in relabel_filter:
            relabel[i] = len(output)
            output.append(out)
    return output, relabel


def cut_model(
    model_file: str,
    input_names: Optional[list],
    output_names: Optional[list],
    subgraph_index: int,
    output_file: str,
) -> None:
    """Cut subgraphs and simplify a given model."""
    model = load(model_file)
    subgraph = model.subgraphs[subgraph_index]
    cut_subgraph(subgraph, input_names, output_names)
    simplify(model)
    save(model, output_file)
