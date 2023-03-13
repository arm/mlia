# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mlia.nn.rewrite.core.utils.utils import load, save


def join_models(input_src, input_dst, output_file, subgraph_src=0, subgraph_dst=0):
    src_model = load(input_src)
    dst_model = load(input_dst)
    src_subgraph = src_model.subgraphs[subgraph_src]
    dst_subgraph = dst_model.subgraphs[subgraph_dst]
    join_subgraphs(src_model, src_subgraph, dst_model, dst_subgraph)
    save(dst_model, output_file)


def join_subgraphs(src_model, src_subgraph, dst_model, dst_subgraph):
    """Copy subgraph src into subgraph dst from model, connecting tensors with the same names"""
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

    assert not (src_to_dst and dst_to_src), (
        "Source and destination subgraphs appear to connect in a loop: %d tensors from src to dst, %d tensors from dst to src"
        % (len(src_to_dst), len(dst_to_src))
    )

    # Relabel matched input/output tensors between graphs
    tensor_relabel = src_to_dst if src_to_dst else dst_to_src

    # Remove matched inputs/outputs as these will now become internal tensors
    if src_to_dst:
        src_subgraph.outputs = [
            o for o in src_subgraph.outputs if not o in tensor_relabel.keys()
        ]
        dst_subgraph.inputs = [
            i for i in dst_subgraph.inputs if not i in tensor_relabel.values()
        ]
    else:
        src_subgraph.inputs = [
            i for i in src_subgraph.inputs if not i in tensor_relabel.keys()
        ]
        dst_subgraph.outputs = [
            o for o in dst_subgraph.outputs if not o in tensor_relabel.values()
        ]

    buffer_relabel = {
        src_subgraph.tensors[i].buffer: dst_subgraph.tensors[o].buffer
        for i, o in tensor_relabel.items()
    }

    used_tensors = [
        t for i, t in enumerate(src_subgraph.tensors) if not i in tensor_relabel
    ]

    used_buffer_ids = [t.buffer for t in used_tensors]

    opcode_data = lambda c: (
        c.builtinCode,
        c.deprecatedBuiltinCode,
        c.customCode,
        c.version,
    )
    opcode_relabel = {
        s: d
        for s in range(len(src_model.operatorCodes))
        for d in range(len(dst_model.operatorCodes))
        if opcode_data(src_model.operatorCodes[s])
        == opcode_data(dst_model.operatorCodes[d])
    }

    # operator order defines execution schedule so must reflect the inputs/outputs dependencies
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
        if not i in buffer_relabel:
            buffer_relabel[i] = len(dst_model.buffers)
            dst_model.buffers.append(src_model.buffers[i])

    for o in src_subgraph.operators:
        o.inputs = [tensor_relabel[t] for t in o.inputs]
        o.outputs = [tensor_relabel[t] for t in o.outputs]
        o.opcodeIndex = opcode_relabel[o.opcodeIndex]

    for t in used_tensors:
        t.buffer = buffer_relabel[t.buffer]

    src_subgraph.inputs = [tensor_relabel[t] for t in src_subgraph.inputs]
    src_subgraph.outputs = [tensor_relabel[t] for t in src_subgraph.outputs]

    dst_subgraph.inputs = list(set(src_subgraph.inputs).union(dst_subgraph.inputs))
    dst_subgraph.outputs = list(set(src_subgraph.outputs).union(dst_subgraph.outputs))


def append_relabel(src, dst, map=None):
    if map is None:
        map = {}
    for i, x in enumerate(src):
        if not i in map:
            map[i] = len(dst)
            dst.append(x)
    return map
