# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model configuration."""
from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List

import numpy as np
import tensorflow as tf

from mlia.core.context import Context
from mlia.nn.tensorflow.optimizations.quantization import dequantize
from mlia.nn.tensorflow.optimizations.quantization import is_quantized
from mlia.nn.tensorflow.optimizations.quantization import QuantizationParameters
from mlia.nn.tensorflow.optimizations.quantization import quantize
from mlia.nn.tensorflow.tflite_convert import convert_to_tflite
from mlia.nn.tensorflow.tflite_graph import load_fb
from mlia.nn.tensorflow.tflite_graph import save_fb
from mlia.nn.tensorflow.utils import check_tflite_datatypes
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.nn.tensorflow.utils import is_saved_model
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


class ModelConfiguration:
    """Base class for model configuration."""

    def __init__(self, model_path: str | Path) -> None:
        """Init model configuration instance."""
        self.model_path = str(model_path)

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        raise NotImplementedError()

    def convert_to_keras(self, keras_model_path: str | Path) -> KerasModel:
        """Convert model to Keras format."""
        raise NotImplementedError()


class KerasModel(ModelConfiguration):
    """Keras model configuration.

    Supports all models supported by Keras API: saved model, H5, HDF5
    """

    def get_keras_model(self) -> tf.keras.Model:
        """Return associated Keras model."""
        try:
            keras_model = tf.keras.models.load_model(self.model_path)
        except OSError as err:
            raise RuntimeError(
                f"Unable to load model content in {self.model_path}. "
                f"Verify that it's a valid model file."
            ) from err

        return keras_model

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        with log_action("Converting Keras to TensorFlow Lite ..."):
            convert_to_tflite(
                self.get_keras_model(),
                quantized,
                input_path=Path(self.model_path),
                output_path=Path(tflite_model_path),
                subprocess=True,
            )

        logger.debug(
            "Model %s converted and saved to %s", self.model_path, tflite_model_path
        )

        return TFLiteModel(tflite_model_path)

    def convert_to_keras(self, keras_model_path: str | Path) -> KerasModel:
        """Convert model to Keras format."""
        return self


TFLiteIODetails = Dict[str, Dict[str, Any]]
TFLiteIODetailsList = List[TFLiteIODetails]
NameToTensorMap = Dict[str, np.ndarray]


class TFLiteModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TensorFlow Lite model configuration."""

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int | None = None,
        num_threads: int | None = None,
    ) -> None:
        """Initiate a TFLite Model."""
        super().__init__(model_path)
        if not num_threads:
            num_threads = None
        if not batch_size:
            try:
                self.interpreter = tf.lite.Interpreter(
                    model_path=self.model_path, num_threads=num_threads
                )
            except ValueError as err:
                raise RuntimeError(
                    f"Unable to load model content in {self.model_path}. "
                    f"Verify that it's a valid model file."
                ) from err
        else:  # if a batch size is specified, modify the TFLite model to use this size
            with tempfile.TemporaryDirectory() as tmp:
                flatbuffer = load_fb(self.model_path)
                for subgraph in flatbuffer.subgraphs:
                    for tensor in list(subgraph.inputs) + list(subgraph.outputs):
                        subgraph.tensors[tensor].shape = np.array(
                            [batch_size] + list(subgraph.tensors[tensor].shape[1:]),
                            dtype=np.int32,
                        )
                tempname = Path(tmp, "rewrite_tmp.tflite")
                save_fb(flatbuffer, tempname)
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(tempname), num_threads=num_threads
                )

        try:
            self.interpreter.allocate_tensors()
        except RuntimeError:
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        details = list(self.input_details) + list(self.output_details)
        self.handle_from_name = {d["name"]: d["index"] for d in details}
        self.shape_from_name = {d["name"]: d["shape"] for d in details}
        self.batch_size = next(iter(self.shape_from_name.values()))[0]

        # Prepare quantization parameters for input and output
        def named_quant_params(
            details: TFLiteIODetailsList,
        ) -> dict[str, QuantizationParameters]:
            return {
                str(detail["name"]): QuantizationParameters(
                    **detail["quantization_parameters"]
                )
                for detail in details
                if TFLiteModel._is_tensor_quantized(detail)
            }

        self._quant_params_input = named_quant_params(self.input_details)
        self._quant_params_output = named_quant_params(self.output_details)

    def __call__(self, named_input: dict) -> NameToTensorMap:
        """Execute the model on one or a batch of named inputs \
            (a dict of name: numpy array)."""
        input_len = next(iter(named_input.values())).shape[0]
        full_steps = input_len // self.batch_size
        remainder = input_len % self.batch_size

        named_ys = defaultdict(list)
        for i in range(full_steps):
            for name, x_batch in named_input.items():
                x_tensor = x_batch[i : i + self.batch_size]  # noqa: E203
                self.interpreter.set_tensor(self.handle_from_name[name], x_tensor)
            self.interpreter.invoke()
            for output_detail in self.output_details:
                named_ys[output_detail["name"]].append(
                    self.interpreter.get_tensor(output_detail["index"])
                )
        if remainder:
            for name, x_batch in named_input.items():
                x_tensor = np.zeros(  # pylint: disable=invalid-name
                    self.shape_from_name[name]
                ).astype(x_batch.dtype)
                x_tensor[:remainder] = x_batch[-remainder:]
                self.interpreter.set_tensor(self.handle_from_name[name], x_tensor)
            self.interpreter.invoke()
            for output_detail in self.output_details:
                named_ys[output_detail["name"]].append(
                    self.interpreter.get_tensor(output_detail["index"])[:remainder]
                )
        return {k: np.concatenate(v) for k, v in named_ys.items()}

    def input_tensors(self) -> list[str]:
        """Return name from input details."""
        return [d["name"] for d in self.input_details]

    def output_tensors(self) -> list[str]:
        """Return name from output details."""
        return [d["name"] for d in self.output_details]

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        return self

    def _tensor_details(
        self, name: str | None = None, idx: int | None = None
    ) -> TFLiteIODetails:
        """Get the details of the tensor by name or index."""
        if idx is not None:
            details = self.interpreter.get_tensor_details()[idx]
            assert details["index"] == idx
        elif name is not None:
            for details_ in self.interpreter.get_tensor_details():
                if name == details_["name"]:
                    details = details_
                    break
            else:
                raise NameError(
                    f"Tensor '{name}' not found in model {self.model_path}."
                )
        else:
            raise ValueError("Either tensor name or index needs to be passed.")

        assert isinstance(details, dict)
        return cast(TFLiteIODetails, details)

    @staticmethod
    def _is_tensor_quantized(details: TFLiteIODetails) -> bool:
        """Use tensor details to check if the corresponding tensor is quantized."""
        quant_params = QuantizationParameters(**details["quantization_parameters"])
        return is_quantized(quant_params)

    def is_tensor_quantized(
        self,
        name: str | None = None,
        idx: int | None = None,
    ) -> bool:
        """Check if the given tensor (identified by name or index) is quantized."""
        details = self._tensor_details(name, idx)
        return self._is_tensor_quantized(details)

    def check_datatypes(self, *allowed_types: type) -> None:
        """Check if the model only has the given allowed datatypes."""
        check_tflite_datatypes(self.model_path, *allowed_types)

    @staticmethod
    def _quant_dequant(
        tensors: NameToTensorMap,
        quant_params: dict[str, QuantizationParameters],
        func: Callable,
    ) -> NameToTensorMap:
        """Quantize/de-quantize tensor using the given parameters and function."""
        return {
            name: (func(tensor, quant_params[name]) if name in quant_params else tensor)
            for name, tensor in tensors.items()
        }

    def dequantize_outputs(self, outputs: NameToTensorMap) -> NameToTensorMap:
        """De-quantize the given model outputs."""
        dequant_outputs = self._quant_dequant(
            outputs, self._quant_params_output, dequantize
        )
        return dequant_outputs

    def quantize_inputs(self, inputs: NameToTensorMap) -> NameToTensorMap:
        """Quantize the given model inputs."""
        quant_inputs = self._quant_dequant(inputs, self._quant_params_input, quantize)
        return quant_inputs


class TfModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TensorFlow model configuration.

    Supports models supported by TensorFlow API (not Keras)
    """

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        convert_to_tflite(
            self.model_path,
            quantized,
            input_path=Path(self.model_path),
            output_path=Path(tflite_model_path),
        )

        return TFLiteModel(tflite_model_path)


def get_model(model: str | Path) -> ModelConfiguration:
    """Return the model object."""
    if is_tflite_model(model):
        return TFLiteModel(model)

    if is_keras_model(model):
        return KerasModel(model)

    if is_saved_model(model):
        return TfModel(model)

    raise ValueError(
        "The input model format is not supported "
        "(supported formats: TensorFlow Lite, Keras, TensorFlow saved model)!"
    )


def get_tflite_model(model: str | Path, ctx: Context) -> TFLiteModel:
    """Convert input model to TensorFlow Lite and returns TFLiteModel object."""
    dst_model_path = ctx.get_model_path("converted_model.tflite")
    src_model = get_model(model)

    return src_model.convert_to_tflite(dst_model_path, quantized=True)


def get_keras_model(model: str | Path, ctx: Context) -> KerasModel:
    """Convert input model to Keras and returns KerasModel object."""
    keras_model_path = ctx.get_model_path("converted_model.h5")
    converted_model = get_model(model)

    return converted_model.convert_to_keras(keras_model_path)
