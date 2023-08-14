# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Collection of Cortex-A operator compatibility information."""
from __future__ import annotations

# pylint: disable=duplicate-code

VERSION = "22.11"
OPERATORS = {
    "builtin_ops": {
        "ABS": {},
        "ADD": {},
        "ARG_MAX": {},
        "ARG_MIN": {},
        "AVERAGE_POOL_2D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "BATCH_MATMUL": {},
        "BATCH_TO_SPACE_ND": {},
        "CAST": {},
        "CONCATENATION": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "CONV_2D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "CONV_3D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "DEPTH_TO_SPACE": {},
        "DEPTHWISE_CONV_2D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "DEQUANTIZE": {},
        "DIV": {},
        "ELU": {},
        "EQUAL": {},
        "EXP": {},
        "EXPAND_DIMS": {},
        "FILL": {},
        "FLOOR": {},
        "FLOOR_DIV": {},
        "FULLY_CONNECTED": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "GATHER": {},
        "GATHER_ND": {},
        "GREATER": {},
        "GREATER_EQUAL": {},
        "HARD_SWISH": {},
        "L2_NORMALIZATION": {},
        "L2_POOL_2D": {},
        "LESS": {},
        "LESS_EQUAL": {},
        "LOCAL_RESPONSE_NORMALIZATION": {},
        "LOG": {},
        "LOGICAL_AND": {},
        "LOGICAL_NOT": {},
        "LOGICAL_OR": {},
        "LOGISTIC": {},
        "LOG_SOFTMAX": {},
        "LSTM": {},
        "MAX_POOL_2D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "TANH",
                "NONE",
            ]
        },
        "MAXIMUM": {},
        "MEAN": {},
        "MINIMUM": {},
        "MIRROR_PAD": {},
        "MUL": {},
        "NEG": {},
        "NOT_EQUAL": {},
        "PACK": {},
        "PAD": {},
        "PADV2": {},
        "PRELU": {},
        "QUANTIZE": {},
        "RANK": {},
        "REDUCE_MAX": {},
        "REDUCE_MIN": {},
        "REDUCE_PROD": {},
        "RELU": {},
        "RELU_N1_TO_1": {},
        "RELU6": {},
        "RESHAPE": {},
        "RESIZE_BILINEAR": {},
        "RESIZE_NEAREST_NEIGHBOR": {},
        "RSQRT": {},
        "SHAPE": {},
        "SIN": {},
        "SOFTMAX": {},
        "SPACE_TO_BATCH_ND": {},
        "SPACE_TO_DEPTH": {},
        "SPLIT": {},
        "SPLIT_V": {},
        "SQRT": {},
        "SQUEEZE": {},
        "STRIDED_SLICE": {},
        "SUB": {},
        "SUM": {},
        "TANH": {},
        "TRANSPOSE": {},
        "TRANSPOSE_CONV": {},
        "UNIDIRECTIONAL_SEQUENCE_LSTM": {},
        "UNPACK": {},
    },
    # CUSTOM OPERATORS
    "custom_ops": {
        "AveragePool3D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "SIGN_BIT",
                "TANH",
                "NONE",
            ]
        },
        "MaxPool3D": {
            "supported_fused_activation": [
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "SIGMOID",
                "SIGN_BIT",
                "TANH",
                "NONE",
            ]
        },
    },
}
