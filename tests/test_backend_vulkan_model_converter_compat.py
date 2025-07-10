# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tflite_compat module."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

from mlia.backend.vulkan_model_converter.compat import NXCompatibilityChecker
from mlia.backend.vulkan_model_converter.compat import NXModelCompatibilityInfo
from mlia.backend.vulkan_model_converter.compat import NXOperatorCompatibilityInfo
from mlia.backend.vulkan_model_converter.compat import VMCCompatibilityLogReader
from mlia.utils.proc import OutputConsumer


# pylint: disable=line-too-long
@pytest.mark.parametrize(
    "vmc_log, expected_ops, expected_errors",
    [
        (
            # CASE1: Framework Ops that don't lower to TOSA at all
            """
<unknown>:0: error: loc("tfl.custom"): failed to legalize operation 'tosa.custom' that was explicitly marked illegal
            """,
            {},
            {
                "tfl.custom": "failed to legalize operation 'tosa.custom' that was explicitly marked illegal"
            },
        ),
        (
            # CASE2: Framework Ops that are supported in the VGF but via generated/substitute shaders
            """
Successfully lowered: tosa.custom at loc("tfl.custom")
Successfully lowered: tosa.max_pool2d at loc("cs_ne_cs_model/quant_max_pooling2d/MaxPool")
            """,
            {
                "cs_ne_cs_model/quant_max_pooling2d/MaxPool": "tosa.max_pool2d",
                "tfl.custom": "tosa.custom",
            },
            {},
        ),
        (
            # CASE3: Framework Ops that successfully lowered to TOSA
            """
Successfully lowered: tfl.pseudo_qconst at loc("inference/coefficients/splat/conv5/convolution")
Successfully lowered: tfl.fully_connected at loc("inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd")
            """,
            {
                "inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd": "tfl.fully_connected",
                "inference/coefficients/splat/conv5/convolution": "tfl.pseudo_qconst",
            },
            {},
        ),
        (
            # CASE3/Fused:
            """
Successfully lowered: tfl.split_v at loc(fused["arm_nss_clampnet_v1_1/split/split", "arm_nss_clampnet_v1_1/split/split1"])
            """,
            {"arm_nss_clampnet_v1_1/split/split": "tfl.split_v"},
            {},
        ),
        (
            # Other errors:
            """
<unknown>:0: error: loc("model/tf.math.multiply_75/Mul1"): failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion
            """,
            {},
            {
                "model/tf.math.multiply_75/Mul1": "failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion"
            },
        ),
    ],
)
def test_vmc_log_parse_line(
    vmc_log: str, expected_ops: list, expected_errors: list
) -> None:
    """ "Test log parsing, mainly to extract compatibility info."""

    reader = VMCCompatibilityLogReader()

    for line in vmc_log.splitlines():
        reader(line.lstrip())

    assert reader.lowered_ops == expected_ops

    assert reader.lowering_errors == expected_errors


def test_vmc_log_parser_overall() -> None:
    """ "Test log parsing, mainly to extract compatibility info."""

    vmc_log = """
    CASE1: Framework Ops that don't lower to TOSA at all
    <unknown>:0: error: loc("tfl.custom"): failed to legalize operation 'tosa.custom' that was explicitly marked illegal

    CASE2: Framework Ops that are supported in the VGF but via generated/substitute shaders
    Successfully lowered: tosa.custom at loc("tfl.custom")
    Successfully lowered: tosa.max_pool2d at loc("cs_ne_cs_model/quant_max_pooling2d/MaxPool")

    CASE3: Framework Ops that successfully lowered to TOSA
    Successfully lowered: tfl.pseudo_qconst at loc("inference/coefficients/splat/conv5/convolution")
    Successfully lowered: tfl.fully_connected at loc("inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd")

    CASE3/Fused:
    Successfully lowered: tfl.split_v at loc(fused["arm_nss_clampnet_v1_1/split/split", "arm_nss_clampnet_v1_1/split/split1"])

    Other errors:
    <unknown>:0: error: loc("model/tf.math.multiply_75/Mul1"): failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion
    """

    reader = VMCCompatibilityLogReader()

    for line in vmc_log.splitlines():
        reader(line.lstrip())

    assert reader.lowered_ops == {
        "arm_nss_clampnet_v1_1/split/split": "tfl.split_v",
        "cs_ne_cs_model/quant_max_pooling2d/MaxPool": "tosa.max_pool2d",
        "inference/coefficients/global/fc2/MatMul;inference/coefficients/global/fc2/Relu;inference/coefficients/global/fc2/BiasAdd": "tfl.fully_connected",
        "inference/coefficients/splat/conv5/convolution": "tfl.pseudo_qconst",
        "tfl.custom": "tosa.custom",
    }

    assert reader.lowering_errors == {
        "tfl.custom": "failed to legalize operation 'tosa.custom' that was explicitly marked illegal",
        "model/tf.math.multiply_75/Mul1": "failed to materialize conversion for result #0 of operation 'tfl.broadcast_to' that remained live after conversion",
    }


def test_parse_loc_simple() -> None:
    """Test parse loc() string, simple case."""
    loc = VMCCompatibilityLogReader().parse_loc(
        'loc("hierarchy/dotted.dashes-semi:location")'
    )
    assert loc == "hierarchy/dotted.dashes-semi:location"


def test_parse_loc_fused() -> None:
    """Test parse loc() string, fused case."""
    loc = VMCCompatibilityLogReader().parse_loc(
        'loc(fused["hierarchy/dotted.dashes-semi:loc", "hierarchy/dotted.dashes-semi:loc1"])'
    )
    assert loc == "hierarchy/dotted.dashes-semi:loc"


@pytest.mark.parametrize(
    "line",
    ["loc(unmatched", "loc(noquotes)", 'loc(fused["op1", "op2")', "foo", 'loc("a b")'],
)
def test_vmc_log_parser_invalid_loc(line: str) -> None:
    """Test log parsing, with invalid syntax."""

    with pytest.raises(Exception, match="Can't find a valid location string"):
        VMCCompatibilityLogReader().parse_loc(line)


def test_checker_calls_vmc_correctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test VMC compatibity check."""

    front_end_output = ["""Successfully lowered: tosa.custom at loc("tfl.custom")"""]
    back_end_output = [
        """<unknown>:0: error: loc("model/tf.math.multiply_75/Mul1"): failed to materialize conversion for result #0 of"""
        + """operation 'tfl.broadcast_to' that remained live after conversion"""
        ""
    ]

    def front_end_call(consumer: OutputConsumer, program: str, *args: str) -> None:
        """Fake Frontend call."""
        if not program.endswith("frontend"):
            pytest.fail("Expected frontend call")
        assert "--experimental-analysis" in args
        output_index = args.index("-o") + 1
        output_path = args[output_index]
        Path(output_path).touch()
        for line in front_end_output:
            consumer(line)

    def back_end_call(consumer: OutputConsumer, program: str, *args: list[str]) -> None:
        """Fake Backend call."""
        if not program.endswith("backend"):
            pytest.fail("Expected backend call")
        assert "--experimental-analysis" in args
        for line in back_end_output:
            consumer(line)

    vmc_commands: list[Callable] = [front_end_call, back_end_call]

    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.conversion.process_command_output",
        lambda cmd, consumers: vmc_commands.pop(0)(consumers[1], *cmd.cmd),
    )

    mock_repo = MagicMock()
    mock_repo.get_backend_settings = MagicMock(return_value=(tmp_path / "backend", {}))
    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.compat.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )

    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.compat.operator_names_to_types",
        MagicMock(return_value={"model/tf.math.multiply_75/Mul1": "MUL"}),
    )

    checker = NXCompatibilityChecker(tmp_path)

    result = checker.check_compatibility(Path("model.tflite"))

    assert len(vmc_commands) == 0
    assert result.dump() == [
        {
            "compat_level": "Non-NX",
            "error": "failed to materialize conversion for result #0 ofoperation "
            "'tfl.broadcast_to' that remained live after conversion",
            "location": "model/tf.math.multiply_75/Mul1",
            "type": "MUL",
        },
        {
            "location": "tfl.custom",
            "compat_level": "Shader",
            "tosa_op": "tosa.custom",
            "placement": "EE",
        },
    ]


def test_nx_compatiblity_info() -> None:
    """Test Neural Accelerator CompatibilityInfo additions."""

    info = NXModelCompatibilityInfo()
    info.add_lowered_to_tosa("model/myloc1/op1", "mytosa_op")
    assert info.dump() == [
        {
            "compat_level": "TOSA",
            "location": "model/myloc1/op1",
            "placement": "NE",
            "tosa_op": "mytosa_op",
        },
    ]

    info.add_lowering_error("model/myloc2/op3", "Can't be lowered")

    assert info.get_records() == [
        NXOperatorCompatibilityInfo(
            location="model/myloc1/op1",
            compat_level="TOSA",
            type=None,
            tosa_op="mytosa_op",
            error=None,
            placement="NE",
        ),
        NXOperatorCompatibilityInfo(
            location="model/myloc2/op3",
            compat_level="Non-NX",
            type=None,
            tosa_op=None,
            error="Can't be lowered",
            placement=None,
        ),
    ]
    assert info.dump() == [
        {
            "location": "model/myloc1/op1",
            "compat_level": "TOSA",
            "tosa_op": "mytosa_op",
            "placement": "NE",
        },
        {
            "location": "model/myloc2/op3",
            "compat_level": "Non-NX",
            "error": "Can't be lowered",
        },
    ]

    info.add_lowered_to_tosa("model/myloc2/shader_op", "tosa.custom")
    assert info.dump() == [
        {
            "location": "model/myloc1/op1",
            "compat_level": "TOSA",
            "tosa_op": "mytosa_op",
            "placement": "NE",
        },
        {
            "location": "model/myloc2/op3",
            "compat_level": "Non-NX",
            "error": "Can't be lowered",
        },
        {
            "location": "model/myloc2/shader_op",
            "compat_level": "Shader",
            "tosa_op": "tosa.custom",
            "placement": "EE",
        },
    ]

    info.add_lowered_to_tosa("model/myloc1/op4", "mytosa_op4")
    assert info.dump() == [
        {
            "location": "model/myloc1/op1",
            "compat_level": "TOSA",
            "tosa_op": "mytosa_op",
            "placement": "NE",
        },
        {
            "location": "model/myloc1/op4",
            "compat_level": "TOSA",
            "tosa_op": "mytosa_op4",
            "placement": "NE",
        },
        {
            "location": "model/myloc2/op3",
            "compat_level": "Non-NX",
            "error": "Can't be lowered",
        },
        {
            "location": "model/myloc2/shader_op",
            "compat_level": "Shader",
            "tosa_op": "tosa.custom",
            "placement": "EE",
        },
    ]
