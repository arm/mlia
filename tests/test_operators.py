# Copyright 2021, Arm Ltd.
"""Tests for advisor module."""
from typing import Any
from typing import Tuple

import pytest
from mlia.config import EthosU55
from mlia.config import TFLiteModel
from mlia.operators import supported_operators


@pytest.mark.parametrize(
    "params, expected_error",
    [
        (
            ["model.tflite", EthosU55()],
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
