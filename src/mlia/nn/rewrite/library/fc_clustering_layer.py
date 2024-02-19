# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Example rewrite with one fully connected clustered layer."""
from typing import Any

from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.fc_layer import get_keras_model


def get_keras_model_clus(input_shape: Any, output_shape: Any) -> keras.Model:
    """Generate TensorFlow Lite model for clustering rewrite."""
    return get_keras_model(input_shape, output_shape)
