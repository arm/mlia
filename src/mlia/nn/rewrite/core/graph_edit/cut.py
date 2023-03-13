# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mlia.nn.rewrite.core.utils.utils import load, save


def cut_subgraph(subgraph, input_tensor_names, output_tensor_names):
    """Change the global inputs and outputs of a graph to the provided named tensors"""

    def tensors_by_name(names):
        seek = frozenset([name.encode("utf-8") for name in names])
        tensors = [
            i for i, tensor in enumerate(subgraph.tensors) if tensor.name in seek
        ]
        return tensors

    if input_tensor_names is not None:
        subgraph.inputs = tensors_by_name(input_tensor_names)
        assert len(subgraph.inputs) == len(
            input_tensor_names
        ), "Expected %d input tensors: %s\nFound: %s" % (
            len(subgraph.inputs),
            ", ".join(input_tensor_names),
            ", ".join(subgraph.tensors[i].name for i in subgraph.inputs),
        )

    if output_tensor_names is not None:
        subgraph.outputs = tensors_by_name(output_tensor_names)
        assert len(subgraph.outputs) == len(
            output_tensor_names
        ), "Expected %d output tensors: %s\nFound: %s" % (
            len(subgraph.outputs),
            ", ".join(output_tensor_names),
            ", ".join(subgraph.tensors[i].name for i in subgraph.outputs),
        )


def simplify(model):
    """Remove any unused operators, tensors and buffers from a model"""
    for s in model.subgraphs:
        simplify_subgraph(s)

    used_buffers = {t.buffer for t in s.tensors for s in model.subgraphs}
    used_buffers = used_buffers.union({m.buffer for m in model.metadata})
    used_buffers.add(
        0
    )  # Buffer zero is always expected to be a zero-sized nullptr buffer by the TFLite runtime
    model.buffers, buf_relabel = filter_relabel(model.buffers, used_buffers)

    for s in model.subgraphs:
        for t in s.tensors:
            t.buffer = buf_relabel[t.buffer]

    for m in model.metadata:
        m.buffer = buf_relabel[m.buffer]


def simplify_subgraph(subgraph):
    requires = defaultdict(set)

    for o, operator in enumerate(subgraph.operators):
        for t in operator.outputs:
            if not t in subgraph.inputs:
                requires[t].add(o)

    op_set, ten_set = find_required(subgraph, requires, subgraph.outputs)

    subgraph.operators, op_relabel = filter_relabel(subgraph.operators, op_set)
    subgraph.tensors, ten_relabel = filter_relabel(subgraph.tensors, ten_set)

    ten_relabel[-1] = -1  # Some files have ops with -1 input tensors; leave unchanged

    for op in subgraph.operators:
        op.inputs = [ten_relabel[t] for t in op.inputs]
        op.outputs = [ten_relabel[t] for t in op.outputs]

    subgraph.inputs = [ten_relabel[t] for t in subgraph.inputs]
    subgraph.outputs = [ten_relabel[t] for t in subgraph.outputs]


def find_required(subgraph, requires, tensors):
    visited_operators = set()
    visited_tensors = set(tensors)
    stop_tensors = set(subgraph.inputs)
    changed = True

    next_tensors = visited_tensors
    while next_tensors:
        loop_tensors = next_tensors
        next_tensors = set()
        for t in loop_tensors:
            candidate_operators = set(requires[t])
            new_operators = candidate_operators - visited_operators
            visited_operators = visited_operators.union(new_operators)
            for op in new_operators:
                candidate_tensors = set(subgraph.operators[op].inputs)
                new_tensors = candidate_tensors - (visited_tensors.union(next_tensors))
                next_tensors = next_tensors.union(new_tensors)
                visited_tensors = visited_tensors.union(candidate_tensors)
                visited_tensors = visited_tensors.union(
                    subgraph.operators[op].outputs
                )  # include stub outputs but do not traverse them
        next_tensors = next_tensors - stop_tensors

    return visited_operators, visited_tensors


def filter_relabel(src, filter):
    relabel = {}
    output = []
    for i, x in enumerate(src):
        if i in filter:
            relabel[i] = len(output)
            output.append(x)
    return output, relabel


def cut_model(model_file, input_names, output_names, subgraph_index, output_file):
    model = load(model_file)
    subgraph = model.subgraphs[subgraph_index]
    cut_subgraph(subgraph, input_names, output_names)
    simplify(model)
    save(model, output_file)
