# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U MLIA module."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.ethos_u.advisor import configure_and_get_ethosu_advisor
from mlia.target.ethos_u.advisor import EthosUInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert EthosUInferenceAdvisor.name() == "ethos_u_inference_advisor"


@pytest.mark.parametrize(
    "optimization_targets, expected_error",
    [
        [
            [
                {
                    "optimization_type": "pruning",
                    "optimization_target": 0.5,
                    "layers_to_optimize": None,
                }
            ],
            pytest.raises(
                Exception,
                match="Only 'rewrite' is supported for TensorFlow Lite files.",
            ),
        ],
        [
            [
                {
                    "optimization_type": "rewrite",
                    "optimization_target": "fully_connected",
                    "layers_to_optimize": [
                        "MobileNet/avg_pool/AvgPool",
                        "MobileNet/fc1/BiasAdd",
                    ],
                }
            ],
            does_not_raise(),
        ],
    ],
)
def test_unsupported_advice_categories(
    tmp_path: Path,
    test_tflite_model: Path,
    optimization_targets: list[dict[str, Any]],
    expected_error: Any,
) -> None:
    """Test that advisor should throw an exception for unsupported categories."""
    with expected_error:
        ctx = ExecutionContext(
            output_dir=tmp_path, advice_category={AdviceCategory.OPTIMIZATION}
        )

        advisor = configure_and_get_ethosu_advisor(
            ctx,
            "ethos-u55-256",
            str(test_tflite_model),
            optimization_targets=optimization_targets,
        )
        advisor.configure(ctx)
