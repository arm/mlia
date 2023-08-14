# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Collection of Cortex-A operator compatibility information."""
from __future__ import annotations

# pylint: disable=duplicate-code

VERSION = "23.05"
SUPPORTED_FUSED_ACTIVATION_FUNCTIONS = [
    "NONE",
    "RELU",
    "RELU6",
    "RELU_N1_TO_1",
    "SIGMOID",
    "TANH",
]
OPERATORS = {
    "builtin_ops": {
        "ABS": {},
        "ADD": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "ARG_MAX": {},
        "ARG_MIN": {},
        "AVERAGE_POOL_2D": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "BATCH_MATMUL": {},
        "BATCH_TO_SPACE_ND": {},
        "CAST": {},
        "CEIL": {},
        "CONCATENATION": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "CONV_2D": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "CONV_3D": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "DEPTH_TO_SPACE": {},
        "DEPTHWISE_CONV_2D": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "DEQUANTIZE": {},
        "DIV": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "ELU": {},
        "EQUAL": {},
        "EXP": {},
        "EXPAND_DIMS": {},
        "FILL": {},
        "FLOOR": {},
        "FLOOR_DIV": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "FULLY_CONNECTED": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "GATHER": {},
        "GATHER_ND": {},
        "GREATER": {},
        "GREATER_EQUAL": {},
        "HARD_SWISH": {},
        "L2_NORMALIZATION": {},
        "L2_POOL_2D": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
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
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "MAXIMUM": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "MEAN": {},
        "MINIMUM": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "MIRROR_PAD": {},
        "MUL": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
        "NEG": {},
        "NOT_EQUAL": {},
        "PACK": {},
        "PAD": {},
        "PADV2": {},
        "PRELU": {},
        "QUANTIZE": {},
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
        "SLICE": {},
        "SOFTMAX": {},
        "SPACE_TO_BATCH_ND": {},
        "SPACE_TO_DEPTH": {},
        "SPLIT": {},
        "SPLIT_V": {},
        "SQRT": {},
        "SQUEEZE": {},
        "STRIDED_SLICE": {},
        "SUB": {"supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS},
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
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
        "MaxPool3D": {
            "supported_fused_activation": SUPPORTED_FUSED_ACTIVATION_FUNCTIONS
        },
    },
}
