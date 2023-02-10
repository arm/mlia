# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/compat."""
from pathlib import Path

import pytest

from mlia.backend.vela.compat import generate_supported_operators_report
from mlia.backend.vela.compat import NpuSupported
from mlia.backend.vela.compat import Operator
from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import supported_operators
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.utils.filesystem import working_directory


@pytest.mark.parametrize(
    "model, expected_ops",
    [
        (
            "test_model.tflite",
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
def test_operators(test_models_path: Path, model: str, expected_ops: Operators) -> None:
    """Test operators function."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")

    operators = supported_operators(
        test_models_path / model, target_config.compiler_options
    )
    for expected, actual in zip(expected_ops.ops, operators.ops):
        # do not compare names as they could be different on each model generation
        assert expected.op_type == actual.op_type
        assert expected.run_on_npu == actual.run_on_npu


def test_generate_supported_operators_report(tmp_path: Path) -> None:
    """Test generating supported operators report."""
    with working_directory(tmp_path):
        generate_supported_operators_report()

        md_file = tmp_path / "SUPPORTED_OPS.md"
        assert md_file.is_file()
        assert md_file.stat().st_size > 0
