# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config module."""
from contextlib import ExitStack as does_not_raise
from typing import Any

import pytest

from mlia.target.cortex_a.config import CortexAConfiguration


@pytest.mark.parametrize(
    "profile_data, expected_error",
    [
        [
            {
                "target": "cortex-a",
                "backend": {"armnn-tflite-delegate": {"version": "23.05"}},
            },
            does_not_raise(),
        ],
        [
            {
                "target": "bad-target",
                "backend": {"armnn-tflite-delegate": {"version": "23.05"}},
            },
            pytest.raises(ValueError, match="Wrong target bad-target"),
        ],
        [
            {
                "target": "cortex-a",
                "backend": {"armnn-tflite-delegate": {"version": None}},
            },
            pytest.raises(
                ValueError,
                match="No version for ArmNN TensorFlow" " Lite delegate specified.",
            ),
        ],
        [
            {
                "target": "cortex-a",
                "backend": {
                    "armnn-tflite-delegate": {"version": "unsupported.version"}
                },
            },
            pytest.raises(ValueError, match="Version 'unsupported.version'"),
        ],
    ],
)
def test_cortex_a_configuration(
    profile_data: dict[str, Any], expected_error: Any
) -> None:
    """Tests CortexAConfiguration"""
    with expected_error:
        cfg = CortexAConfiguration(**profile_data)
        cfg.verify()
