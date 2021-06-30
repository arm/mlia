"""Tests for advisor module."""
from typing import Tuple

import pytest
from mlia.config import EthosU55
from mlia.config import TFLiteModel
from mlia.operators import supported_operators


@pytest.mark.parametrize(
    "params", [["model.tflite", EthosU55()], [TFLiteModel("model.tflite"), "ethos-u55"]]
)
def test_supported_operators_wrong_params(params: Tuple) -> None:
    """Test supported_operators should fail for wrong params."""
    with pytest.raises(Exception):
        model, device = params
        supported_operators(model, device)
