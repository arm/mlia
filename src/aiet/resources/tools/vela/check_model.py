# SPDX-FileCopyrightText: Copyright 2020, 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Check if a TFLite model file is Vela-optimised."""
import struct
from pathlib import Path

from ethosu.vela.tflite.Model import Model

from aiet.cli.common import InvalidTFLiteFileError
from aiet.cli.common import ModelOptimisedException
from aiet.utils.fs import read_file_as_bytearray


def get_model_from_file(input_model_file: Path) -> Model:
    """Generate Model instance from TFLite file using flatc generated code."""
    buffer = read_file_as_bytearray(input_model_file)
    try:
        model = Model.GetRootAsModel(buffer, 0)
    except (TypeError, RuntimeError, struct.error) as tflite_error:
        raise InvalidTFLiteFileError(
            f"Error reading in model from {input_model_file}."
        ) from tflite_error
    return model


def is_vela_optimised(tflite_model: Model) -> bool:
    """Return True if 'ethos-u' custom operator found in the Model."""
    operators = get_operators_from_model(tflite_model)

    custom_codes = get_custom_codes_from_operators(operators)

    return check_custom_codes_for_ethosu(custom_codes)


def get_operators_from_model(tflite_model: Model) -> list:
    """Return list of the unique operator codes used in the Model."""
    return [
        tflite_model.OperatorCodes(index)
        for index in range(tflite_model.OperatorCodesLength())
    ]


def get_custom_codes_from_operators(operators: list) -> list:
    """Return list of each operator's CustomCode() strings, if they exist."""
    return [
        operator.CustomCode()
        for operator in operators
        if operator.CustomCode() is not None
    ]


def check_custom_codes_for_ethosu(custom_codes: list) -> bool:
    """Check for existence of ethos-u string in the custom codes."""
    return any(
        custom_code_name.decode("utf-8") == "ethos-u"
        for custom_code_name in custom_codes
    )


def check_model(tflite_file_name: str) -> None:
    """Raise an exception if model in given file is Vela optimised."""
    tflite_path = Path(tflite_file_name)

    tflite_model = get_model_from_file(tflite_path)

    if is_vela_optimised(tflite_model):
        raise ModelOptimisedException(
            f"TFLite model in {tflite_file_name} is already "
            f"vela optimised ('ethos-u' custom op detected)."
        )

    print(
        f"TFLite model in {tflite_file_name} is not vela optimised "
        f"('ethos-u' custom op not detected)."
    )
