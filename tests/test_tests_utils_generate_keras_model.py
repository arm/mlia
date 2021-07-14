# Copyright 2021, Arm Ltd.
"""Test for module tests/utils/generate_keras_model."""
from tests.utils.generate_keras_model import generate_keras_model


def test_model_generation() -> None:
    """Test for test model generation."""
    generated_model = generate_keras_model()

    assert generated_model.get_layer("conv1")
    assert generated_model.get_layer("conv2")
