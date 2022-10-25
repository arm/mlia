# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for TensorFlow Lite graphs."""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import visualize


def _enum_from_class(cls: Any) -> Any:
    """Create an enum from the public class variables."""
    return enum.Enum(
        cls.__name__,
        {key: value for key, value in vars(cls).items() if not key.startswith("_")},
    )


TFL_TYPE = _enum_from_class(schema_fb.TensorType)
TFL_OP = _enum_from_class(schema_fb.BuiltinOperator)
TFL_ACTIVATION_FUNCTION = _enum_from_class(schema_fb.ActivationFunctionType)


def _ascii_list_to_string(ascii_list: list[int]) -> str:
    return "".join(chr(i) for i in ascii_list)


@dataclass
class TensorInfo:
    """Collection of tensor information parsed from a TensorFlow Lite file."""

    name: str
    type: str
    shape: tuple | list
    is_variable: bool

    def __str__(self) -> str:
        """Create a text represenation of this TensorInfo instance."""
        return f"{self.name}: {self.type}, {self.shape}, is_variable={self.is_variable}"

    def __repr__(self) -> str:
        """Convert this instance to JSON."""
        return json.dumps(vars(self))

    @classmethod
    def from_dict(cls, tensor: dict[str, Any]) -> TensorInfo:
        """
        Create a new instance from a dictionary.

        The expected dict is the one contained in the dict returned by
        visualize.CreateDictFromFlatbuffer().
        """
        return TensorInfo(
            _ascii_list_to_string(tensor["name"]),
            TFL_TYPE(tensor["type"]).name,
            tensor["shape"],
            tensor["is_variable"],
        )


@dataclass
class Op:
    """
    Representation of an operator from a TensorFlow Lite file.

    E.g. collects the operator type, input/output tensors etc.
    """

    type: str
    builtin_options: dict
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    custom_type: str | None = None

    def __post_init__(self) -> None:
        """Convert the builtin option 'fused_activation_function' to string."""
        if "fused_activation_function" in self.builtin_options:
            # Convert the fused activation function ID to a string
            self.builtin_options["fused_activation_function"] = TFL_ACTIVATION_FUNCTION(
                self.builtin_options["fused_activation_function"]
            ).name

    def __str__(self) -> str:
        """Create a text represenation of this Op instance."""
        return f"""{self.type}
    builtin_options: {self.builtin_options}
    inputs: {self.inputs}
    outputs: {self.outputs}"""

    @property
    def is_custom(self) -> bool:
        """Check if this Op is a custom operator."""
        return self.type == cast(str, TFL_OP.CUSTOM.name)

    @classmethod
    def from_model_info(cls, oper: dict, graph: dict, model: dict) -> Op:
        """Create a new Op from the model information."""
        op_code_idx = oper["opcode_index"]
        op_code_obj = model["operator_codes"][op_code_idx]
        op_code = max(
            op_code_obj["builtin_code"], op_code_obj["deprecated_builtin_code"]
        )
        custom_code = op_code_obj.get("custom_code")
        return cls(
            type=cast(str, TFL_OP(op_code).name),
            builtin_options=oper["builtin_options"] if oper["builtin_options"] else {},
            inputs=[
                TensorInfo.from_dict(graph["tensors"][idx]) for idx in oper["inputs"]
            ],
            outputs=[
                TensorInfo.from_dict(graph["tensors"][idx]) for idx in oper["outputs"]
            ],
            custom_type=_ascii_list_to_string(custom_code) if custom_code else None,
        )


def load_tflite(file: Path) -> bytes:
    """Load a TensorFlow Lite file from disk."""
    return file.read_bytes()


def parse_subgraphs(tflite_file: Path) -> list[list[Op]]:
    """Load the TensorFlow Lite file and parse the subgraphs."""
    tflite_model = load_tflite(tflite_file)
    model = cast(dict, visualize.CreateDictFromFlatbuffer(tflite_model))
    assert isinstance(model, dict)

    graphs = [
        [Op.from_model_info(oper, g, model) for oper in g["operators"]]
        for g in model["subgraphs"]
    ]

    return graphs
