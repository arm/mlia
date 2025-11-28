# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A operator compatibility."""
from pathlib import Path

import pytest
import tf_keras as keras

from mlia.nn.tensorflow.tflite_convert import convert_to_tflite_bytes
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.target.cortex_a.operators import get_cortex_a_compatibility_info
from mlia.target.cortex_a.operators import Operator
from mlia.target.cortex_a.operators import TFL_ACTIVATION_FUNCTION


def check_get_cortex_a_compatibility_info(
    model_path: Path,
    expected_success: bool,
) -> None:
    """Check the function 'get_cortex_a_compatibility_info'."""
    compat_info = get_cortex_a_compatibility_info(
        model_path, CortexAConfiguration.load_profile("cortex-a")
    )
    assert isinstance(compat_info, CortexACompatibilityInfo)
    assert expected_success == compat_info.is_cortex_a_compatible
    assert compat_info.operators
    for oper in compat_info.operators:
        assert oper.name
        assert oper.location
        assert (
            compat_info.get_support_type(oper) in CortexACompatibilityInfo.SupportType
        )


def test_get_cortex_a_compatibility_info_compatible(
    test_tflite_model: Path,
) -> None:
    """Test a fully compatible TensorFlow Lite model."""
    check_get_cortex_a_compatibility_info(test_tflite_model, expected_success=True)


def test_get_cortex_a_compatibility_info_not_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct and test a NOT fully compatible TensorFlow Lite model."""
    keras_model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), batch_size=1, name="input"),
            keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="softmax", name="conv1"
            ),
            keras.layers.LeakyReLU(),
        ]
    )
    keras_model.compile(optimizer="sgd", loss="mean_squared_error")
    tflite_model = convert_to_tflite_bytes(keras_model, quantized=False)

    monkeypatch.setattr(
        "mlia.nn.tensorflow.tflite_graph.load_tflite", lambda _p: tflite_model
    )
    check_get_cortex_a_compatibility_info(
        Path("NOT_USED_BECAUSE_OF_MOCKING"), expected_success=False
    )


def test_cortex_a_compatibility_to_standardized_output(tmp_path: Path) -> None:
    """Test conversion of CortexACompatibilityInfo to standardized output."""
    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    # Create test operators
    ops = [
        Operator(
            name="CONV_2D",
            location="subgraph:0,oper:0",
            activation_func=TFL_ACTIVATION_FUNCTION.NONE,
        ),
        Operator(
            name="ADD",
            location="subgraph:0,oper:1",
            activation_func=TFL_ACTIVATION_FUNCTION.RELU,
        ),
        Operator(
            name="CUSTOM",
            location="subgraph:0,oper:2",
            activation_func=TFL_ACTIVATION_FUNCTION.NONE,
            custom_name="MyCustomOp",
        ),
    ]

    compat_info = CortexACompatibilityInfo(ops, "23.05")
    standardized_output = compat_info.to_standardized_output(
        model_path=model_file,
        target_config={"cpu": "cortex-a78"},
    )
    output = standardized_output.to_dict()

    # Verify structure
    assert "schema_version" in output
    assert output["schema_version"] == "1.0.0"
    assert "backends" in output
    assert "target" in output
    assert "model" in output
    assert "context" in output
    assert "results" in output

    # Verify backend
    backends = output["backends"]
    assert len(backends) == 1
    backend = backends[0]
    assert backend["name"] == "Arm NN TensorFlow Lite Delegate"
    assert backend["version"] == "23.05"

    # Verify target
    target = output["target"]
    assert target["profile_name"] == "cortex-a78"
    assert target["target_type"] == "cpu"

    # Verify results
    results = output["results"]
    assert len(results) == 1
    result = results[0]
    assert result["kind"] == "compatibility"

    # Verify checks and entities
    assert "checks" in result
    assert "entities" in result
    checks = result["checks"]
    entities = result["entities"]

    assert len(checks) == 3  # One check per operator
    assert len(entities) == 3  # One entity per operator

    # Verify first operator (CONV_2D - compatible)
    assert entities[0]["name"] == "CONV_2D"
    assert entities[0]["scope"] == "operator"
    assert entities[0]["placement"] == "cpu"
    assert checks[0]["status"] == "pass"

    # Verify third operator (CUSTOM - not supported)
    assert entities[2]["name"] == "CUSTOM - 'MyCustomOp'"
    assert entities[2]["placement"] == "unsupported"
    assert checks[2]["status"] == "fail"


def test_cortex_a_compatibility_to_standardized_output_all_compatible(
    tmp_path: Path,
) -> None:
    """Test conversion when all operators are compatible."""
    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    # Create all compatible operators
    ops = [
        Operator(
            name="CONV_2D",
            location="subgraph:0,oper:0",
            activation_func=TFL_ACTIVATION_FUNCTION.RELU,
        ),
        Operator(
            name="DEPTHWISE_CONV_2D",
            location="subgraph:0,oper:1",
            activation_func=TFL_ACTIVATION_FUNCTION.RELU6,
        ),
    ]

    compat_info = CortexACompatibilityInfo(ops, "23.05")
    standardized_output = compat_info.to_standardized_output(
        model_path=model_file,
        target_config={"cpu": "cortex-a55"},
    )
    output = standardized_output.to_dict()

    # Verify overall status
    results = output["results"]
    result = results[0]
    assert result["status"] == "ok"

    # Verify all checks are PASS
    checks = result["checks"]
    entities = result["entities"]
    assert len(checks) == 2
    assert len(entities) == 2
    assert all(check["status"] == "pass" for check in checks)
    assert all(entity["placement"] == "cpu" for entity in entities)
