# Copyright 2021, Arm Ltd.
"""Tests for advisor module."""
from typing import Any
from typing import Tuple

import pytest
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.operators import supported_operators
from mlia.nn.tensorflow.config import TFLiteModel


@pytest.mark.parametrize(
    "params, expected_error",
    [
        (
            ["model.tflite", EthosUConfiguration(target="U55-256")],
            pytest.raises(Exception, match="Unsupported model configuration"),
        ),
        (
            [TFLiteModel("model.tflite"), "ethos-u55"],
            pytest.raises(Exception, match="Unsupported ip configuration"),
        ),
    ],
)
def test_supported_operators_wrong_params(params: Tuple, expected_error: Any) -> None:
    """Test supported_operators should fail for wrong params."""
    with expected_error:
        model, device = params
        supported_operators(model, device)
