# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Join module."""
from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import ModelT
from tensorflow.lite.python.schema_py_generated import OperatorCodeT
from tensorflow.lite.python.schema_py_generated import SubGraphT

from mlia.nn.rewrite.core.utils.utils import load
from mlia.nn.rewrite.core.utils.utils import save

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def join_models(
    input_src: str | Path,
    input_dst: str | Path,
    output_file: str | Path,
    subgraph_src: int = 0,
    subgraph_dst: int = 0,
) -> None:
    """Join two models and save the result into a given model file path."""
    src_model = load(input_src)
    dst_model = load(input_dst)
    src_subgraph = src_model.subgraphs[subgraph_src]
    dst_subgraph = dst_model.subgraphs[subgraph_dst]
    join_subgraphs(src_model, src_subgraph, dst_model, dst_subgraph)
    save(dst_model, output_file)


def join_subgraphs(
    src_model: ModelT,
    src_subgraph: SubGraphT,
    dst_model: ModelT,
    dst_subgraph: SubGraphT,
) -> None:
    """Join two subgraphs, connecting tensors with the same names."""
    # Find inputs that match outputs in the other graph and vice versa
    dst_to_src = {
        i: o
        for i in src_subgraph.inputs
        for o in dst_subgraph.outputs
        if src_subgraph.tensors[i].name == dst_subgraph.tensors[o].name
    }

    src_to_dst = {
        o: i
        for i in dst_subgraph.inputs
        for o in src_subgraph.outputs
        if dst_subgraph.tensors[i].name == src_subgraph.tensors[o].name
    }

    assert not (
        src_to_dst and dst_to_src
    ), f"Source and destination subgraphs appear to connect in a loop: \
        {len(src_to_dst)} tensors from src to dst, {len(dst_to_src)} \
        tensors from dst to src"

    # Relabel matched input/output tensors between graphs
    tensor_relabel = src_to_dst if src_to_dst else dst_to_src

    # Remove matched inputs/outputs as these will now become internal tensors
    if src_to_dst:
        src_subgraph.outputs = [
            output
            for output in src_subgraph.outputs
            if output not in tensor_relabel.keys()
        ]
        dst_subgraph.inputs = [
            input
            for input in dst_subgraph.inputs
            if input not in tensor_relabel.values()
        ]
    else:
        src_subgraph.inputs = [
            input for input in src_subgraph.inputs if input not in tensor_relabel.keys()
        ]
        dst_subgraph.outputs = [
            output
            for output in dst_subgraph.outputs
            if output not in tensor_relabel.values()
        ]

    buffer_relabel = {
        src_subgraph.tensors[input].buffer: dst_subgraph.tensors[output].buffer
        for input, output in tensor_relabel.items()
    }

    used_tensors = [
        tensor
        for i, tensor in enumerate(src_subgraph.tensors)
        if i not in tensor_relabel
    ]

    used_buffer_ids = [tensor.buffer for tensor in used_tensors]

    def opcode_data(code: OperatorCodeT) -> tuple:
        return (
            code.builtinCode,
            code.deprecatedBuiltinCode,
            code.customCode,
            code.version,
        )

    opcode_relabel = {
        s: d
        for s in range(len(src_model.operatorCodes))
        for d in range(len(dst_model.operatorCodes))
        if opcode_data(src_model.operatorCodes[s])
        == opcode_data(dst_model.operatorCodes[d])
    }

    # operator order defines execution schedule so must reflect
    # the inputs/outputs dependencies
    if dst_to_src:
        dst_subgraph.operators += src_subgraph.operators
    else:
        dst_subgraph.operators = src_subgraph.operators + dst_subgraph.operators

    append_relabel(src_subgraph.tensors, dst_subgraph.tensors, tensor_relabel)
    append_relabel(src_model.operatorCodes, dst_model.operatorCodes, opcode_relabel)

    tensor_relabel[
        -1
    ] = -1  # Some files have ops with -1 input tensors; leave unchanged

    for i in used_buffer_ids:
        if i not in buffer_relabel:
            buffer_relabel[i] = len(dst_model.buffers)
            dst_model.buffers.append(src_model.buffers[i])

    for operator in src_subgraph.operators:
        operator.inputs = [tensor_relabel[tensor] for tensor in operator.inputs]
        operator.outputs = [tensor_relabel[tensor] for tensor in operator.outputs]
        operator.opcodeIndex = opcode_relabel[operator.opcodeIndex]

    for tensor in used_tensors:
        tensor.buffer = buffer_relabel[tensor.buffer]

    src_subgraph.inputs = [tensor_relabel[t] for t in src_subgraph.inputs]
    src_subgraph.outputs = [tensor_relabel[t] for t in src_subgraph.outputs]

    dst_subgraph.inputs = list(set(src_subgraph.inputs).union(dst_subgraph.inputs))
    dst_subgraph.outputs = list(set(src_subgraph.outputs).union(dst_subgraph.outputs))


def append_relabel(src: list, dst: list, operator_map: dict) -> None:
    """Update the operator map over relabeled tensors in a subgraph."""
    if operator_map is None:
        raise ValueError("The input operator map cannot be None!")

    for i, x in enumerate(src):  # pylint: disable=invalid-name
        if i not in operator_map:
            operator_map[i] = len(dst)
            dst.append(x)
