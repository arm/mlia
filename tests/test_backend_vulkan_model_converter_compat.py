# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tflite_compat module."""
from __future__ import annotations

import pytest

from mlia.backend.vulkan_model_converter.compat import VMCCompatibilityLogReader


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
    """ "Test log parsing, with invalid syntax."""

    with pytest.raises(Exception, match="Can't find a valid location string"):
        VMCCompatibilityLogReader().parse_loc(line)
