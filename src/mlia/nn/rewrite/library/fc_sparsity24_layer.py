# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PLACEHOLDER for example rewrite with one fully connected 2:4 sparsity layer."""
from typing import Any

from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from .fc_layer import get_keras_model


def get_keras_model24(input_shape: Any, output_shape: Any) -> keras.Model:
    """Generate TensorFlow Lite model for rewrite."""
    model = get_keras_model(input_shape, output_shape)
    return model
