# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-outer-name,no-self-use
"""Module for testing check_model.py script."""
from pathlib import Path
from typing import Any

import pytest
from ethosu.vela.tflite.Model import Model
from ethosu.vela.tflite.OperatorCode import OperatorCode

from aiet.cli.common import InvalidTFLiteFileError
from aiet.cli.common import ModelOptimisedException
from aiet.resources.tools.vela.check_model import check_custom_codes_for_ethosu
from aiet.resources.tools.vela.check_model import check_model
from aiet.resources.tools.vela.check_model import get_custom_codes_from_operators
from aiet.resources.tools.vela.check_model import get_model_from_file
from aiet.resources.tools.vela.check_model import get_operators_from_model
from aiet.resources.tools.vela.check_model import is_vela_optimised


@pytest.fixture(scope="session")
def optimised_tflite_model(
    optimised_input_model_file: Path,
) -> Model:
    """Return Model instance read from a Vela-optimised TFLite file."""
    return get_model_from_file(optimised_input_model_file)


@pytest.fixture(scope="session")
def non_optimised_tflite_model(
    non_optimised_input_model_file: Path,
) -> Model:
    """Return Model instance read from a Vela-optimised TFLite file."""
    return get_model_from_file(non_optimised_input_model_file)


class TestIsVelaOptimised:
    """Test class for is_vela_optimised() function."""

    def test_return_true_when_input_is_optimised(
        self,
        optimised_tflite_model: Model,
    ) -> None:
        """Verify True returned when input is optimised model."""
        output = is_vela_optimised(optimised_tflite_model)

        assert output is True

    def test_return_false_when_input_is_not_optimised(
        self,
        non_optimised_tflite_model: Model,
    ) -> None:
        """Verify False returned when input is non-optimised model."""
        output = is_vela_optimised(non_optimised_tflite_model)

        assert output is False


def test_get_operator_list_returns_correct_instances(
    optimised_tflite_model: Model,
) -> None:
    """Verify list of OperatorCode instances returned by get_operator_list()."""
    operator_list = get_operators_from_model(optimised_tflite_model)

    assert all(isinstance(operator, OperatorCode) for operator in operator_list)


class TestGetCustomCodesFromOperators:
    """Test the get_custom_codes_from_operators() function."""

    def test_returns_empty_list_when_input_operators_have_no_custom_codes(
        self, monkeypatch: Any
    ) -> None:
        """Verify function returns empty list when operators have no custom codes."""
        # Mock OperatorCode.CustomCode() function to return None
        monkeypatch.setattr(
            "ethosu.vela.tflite.OperatorCode.OperatorCode.CustomCode", lambda _: None
        )

        operators = [OperatorCode()] * 3

        custom_codes = get_custom_codes_from_operators(operators)

        assert custom_codes == []

    def test_returns_custom_codes_when_input_operators_have_custom_codes(
        self, monkeypatch: Any
    ) -> None:
        """Verify list of bytes objects returned representing the CustomCodes."""
        # Mock OperatorCode.CustomCode() function to return a byte string
        monkeypatch.setattr(
            "ethosu.vela.tflite.OperatorCode.OperatorCode.CustomCode",
            lambda _: b"custom-code",
        )

        operators = [OperatorCode()] * 3

        custom_codes = get_custom_codes_from_operators(operators)

        assert custom_codes == [b"custom-code", b"custom-code", b"custom-code"]


@pytest.mark.parametrize(
    "custom_codes, expected_output",
    [
        ([b"ethos-u", b"something else"], True),
        ([b"custom-code-1", b"custom-code-2"], False),
    ],
)
def test_check_list_for_ethosu(custom_codes: list, expected_output: bool) -> None:
    """Verify function detects 'ethos-u' bytes in the input list."""
    output = check_custom_codes_for_ethosu(custom_codes)
    assert output is expected_output


class TestGetModelFromFile:
    """Test the get_model_from_file() function."""

    def test_error_raised_when_input_is_invalid_model_file(
        self,
        invalid_input_model_file: Path,
    ) -> None:
        """Verify error thrown when an invalid model file is given."""
        with pytest.raises(InvalidTFLiteFileError):
            get_model_from_file(invalid_input_model_file)

    def test_model_instance_returned_when_input_is_valid_model_file(
        self,
        optimised_input_model_file: Path,
    ) -> None:
        """Verify file is read successfully and returns model instance."""
        tflite_model = get_model_from_file(optimised_input_model_file)

        assert isinstance(tflite_model, Model)


class TestCheckModel:
    """Test the check_model() function."""

    def test_check_model_with_non_optimised_input(
        self,
        non_optimised_input_model_file: Path,
    ) -> None:
        """Verify no error occurs for a valid input file."""
        check_model(non_optimised_input_model_file)

    def test_check_model_with_optimised_input(
        self,
        optimised_input_model_file: Path,
    ) -> None:
        """Verify that the right exception is raised with already optimised input."""
        with pytest.raises(ModelOptimisedException):
            check_model(optimised_input_model_file)

    def test_check_model_with_invalid_input(
        self,
        invalid_input_model_file: Path,
    ) -> None:
        """Verify that an exception is raised with invalid input."""
        with pytest.raises(Exception):
            check_model(invalid_input_model_file)
