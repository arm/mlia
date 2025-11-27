# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/compat."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mlia.core.output_schema as schema
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela.compat import generate_supported_operators_report
from mlia.backend.vela.compat import get_vela
from mlia.backend.vela.compat import NpuSupported
from mlia.backend.vela.compat import Operator
from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import supported_operators
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.utils.filesystem import working_directory
from tests.conftest import TEST_MODEL_TFLITE_INT8_FILE


def replace_get_vela_with_mock(
    monkeypatch: pytest.MonkeyPatch, mock: MagicMock | None
) -> None:
    """Replace Vela with mock."""
    monkeypatch.setattr(
        "mlia.backend.vela.compat.get_vela",
        MagicMock(return_value=mock),
    )


@pytest.mark.parametrize(
    "name, op_type, npu_supported",
    [
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(False, [("CPU only operator", "")]),
        ),
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(True, []),
        ),
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(False, [("Other reason", "")]),
        ),
    ],
)
def test_operator(name: str, op_type: str, npu_supported: NpuSupported) -> None:
    """Test Operator class."""
    operator = Operator(name, op_type, npu_supported)
    cpu_only = not npu_supported.supported and npu_supported.reasons == [
        ("CPU only operator", "")
    ]
    assert operator.cpu_only == cpu_only


@pytest.mark.parametrize(
    "ops",
    [
        [
            Operator(
                name="sequential/conv1/Relu;sequential/conv1/BiasAdd;"
                "sequential/conv2/Conv2D;sequential/conv1/Conv2D",
                op_type="CONV_2D",
                run_on_npu=NpuSupported(supported=True, reasons=[]),
            ),
            Operator(
                name="sequential/conv2/Relu;sequential/conv2/BiasAdd;"
                "sequential/conv2/Conv2D",
                op_type="CONV_2D",
                run_on_npu=NpuSupported(supported=True, reasons=[]),
            ),
            Operator(
                name="sequential/max_pooling2d/MaxPool",
                op_type="MAX_POOL_2D",
                run_on_npu=NpuSupported(supported=False, reasons=[]),
            ),
        ],
        [],
    ],
)
def test_operators(ops: list[Operator]) -> None:
    """Test operators function."""
    operators = Operators(ops)

    total_ops = len(ops)
    npu_supported_ops = sum(op.run_on_npu.supported for op in ops)

    assert operators.total_number == total_ops
    assert operators.npu_supported_number == npu_supported_ops

    if total_ops > 0:
        assert operators.npu_supported_ratio == npu_supported_ops / total_ops

    assert operators.npu_unsupported_ratio == 1 - operators.npu_supported_ratio


@pytest.mark.parametrize(
    "model, expected_ops",
    [
        (
            TEST_MODEL_TFLITE_INT8_FILE,
            Operators(
                ops=[
                    Operator(
                        name="sequential/conv1/Relu;sequential/conv1/BiasAdd;"
                        "sequential/conv2/Conv2D;sequential/conv1/Conv2D",
                        op_type="CONV_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="sequential/conv2/Relu;sequential/conv2/BiasAdd;"
                        "sequential/conv2/Conv2D",
                        op_type="CONV_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="sequential/max_pooling2d/MaxPool",
                        op_type="MAX_POOL_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="sequential/flatten/Reshape",
                        op_type="RESHAPE",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="Identity",
                        op_type="FULLY_CONNECTED",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                ]
            ),
        )
    ],
)
def test_supported_operators(
    test_models_path: Path, model: str, expected_ops: Operators
) -> None:
    """Test operators function."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")

    try:
        operators = supported_operators(
            test_models_path / model, target_config.compiler_options
        )
        for expected, actual in zip(expected_ops.ops, operators.ops):
            # do not compare names as they could be different on each model generation
            assert expected.op_type == actual.op_type
            assert expected.run_on_npu == actual.run_on_npu
    except BackendUnavailableError:
        # If Vela is not available, the test should pass (expected behavior)
        pytest.skip("Vela backend not available, skipping operators test")


def test_generate_supported_operators_report(tmp_path: Path) -> None:
    """Test generating supported operators report."""
    try:
        with working_directory(tmp_path):
            generate_supported_operators_report()

            md_file = tmp_path / "SUPPORTED_OPS.md"
            assert md_file.is_file()
            assert md_file.stat().st_size > 0
    except BackendUnavailableError:
        # If Vela is not available, the test should pass (expected behavior)
        pytest.skip(
            "Vela backend not available, skipping supported operators report test"
        )


def test_compatibility_check_should_fail_if_checker_not_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path | Path
) -> None:
    """Test that compatibility check should fail if Vela is not available."""
    replace_get_vela_with_mock(monkeypatch, None)

    with working_directory(tmp_path):
        with pytest.raises(
            BackendUnavailableError, match="Backend vela is not available"
        ):
            generate_supported_operators_report()


def test_compatibility_check_should_fail_if_checker_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path | Path
) -> None:
    """Test that compatibility check should fail if Vela checker returns False."""
    # Mock get_vela to return False directly
    monkeypatch.setattr(
        "mlia.backend.vela.compat.get_vela",
        MagicMock(return_value=False),
    )

    with working_directory(tmp_path):
        with pytest.raises(
            BackendUnavailableError, match="Backend vela is not available"
        ):
            generate_supported_operators_report()


def test_get_vela_returns_availability_status() -> None:
    """Test that get_vela returns the correct availability status."""
    # The function should return True if ethosu.vela is available, False otherwise
    result = get_vela()
    # The result should be a boolean indicating vela availability
    assert isinstance(result, bool)


def test_operators_to_standardized_output(tmp_path: Path) -> None:
    """Test conversion of Operators to standardized output."""
    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    ops = [
        Operator(
            name="conv1",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(supported=True, reasons=[]),
        ),
        Operator(
            name="conv2",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(
                supported=False, reasons=[("CPU only operator", "")]
            ),
        ),
        Operator(
            name="pool1",
            op_type="MAX_POOL_2D",
            run_on_npu=NpuSupported(
                supported=False,
                reasons=[("Constraint failed", "Invalid tensor shape")],
            ),
        ),
    ]

    operators = Operators(ops)
    output: dict = operators.to_standardized_output(
        model_path=model_file,
        target_config={"target": "ethos-u55", "mac": 256},
    )

    # Verify structure
    assert "schema_version" in output  # pylint: disable=unsupported-membership-test
    assert output["schema_version"] == schema.SCHEMA_VERSION
    assert "backends" in output  # pylint: disable=unsupported-membership-test
    assert "target" in output  # pylint: disable=unsupported-membership-test
    assert "model" in output  # pylint: disable=unsupported-membership-test
    assert "context" in output  # pylint: disable=unsupported-membership-test
    assert "results" in output  # pylint: disable=unsupported-membership-test

    # Verify backend
    backends = output["backends"]  # pylint: disable=unsupported-membership-test
    assert len(backends) == 1
    backend = backends[0]
    assert backend["name"] == "Vela Compiler"
    assert "version" in backend

    # Verify result
    results = output["results"]  # pylint: disable=unsupported-membership-test
    assert len(results) == 1
    result = results[0]
    assert result["kind"] == "compatibility"
    assert result["status"] == "partial"  # Some supported, some not

    # Verify checks and entities
    assert "checks" in result
    assert "entities" in result
    checks = result["checks"]
    entities = result["entities"]

    assert len(checks) == 3  # One check per operator
    assert len(entities) == 3  # One entity per operator

    # Verify first operator (supported)
    assert entities[0]["name"] == "conv1"
    assert entities[0]["scope"] == "operator"
    assert entities[0]["placement"] == "npu"
    assert checks[0]["status"] == "pass"

    # Verify second operator (CPU only)
    assert entities[1]["name"] == "conv2"
    assert entities[1]["placement"] == "cpu"
    assert checks[1]["status"] == "fail"
    assert "reasons" in checks[1]["details"]

    # Verify third operator (constraint failed)
    assert entities[2]["name"] == "pool1"
    assert entities[2]["placement"] == "cpu"
    assert checks[2]["status"] == "fail"
    assert "reasons" in checks[2]["details"]


def test_operators_to_standardized_output_all_supported(tmp_path: Path) -> None:
    """Test conversion when all operators are supported."""
    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    ops = [
        Operator(
            name="conv1",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(supported=True, reasons=[]),
        ),
        Operator(
            name="conv2",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(supported=True, reasons=[]),
        ),
    ]

    operators = Operators(ops)
    output: dict = operators.to_standardized_output(
        model_path=model_file,
    )

    results = output["results"]  # pylint: disable=unsupported-membership-test
    assert len(results) == 1
    result = results[0]
    assert result["status"] == "ok"  # All supported
