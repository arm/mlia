# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A MLIA module."""
from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.common.optimization import _DEFAULT_OPTIMIZATION_TARGETS
from mlia.target.cortex_a.advisor import configure_and_get_cortexa_advisor
from mlia.target.cortex_a.advisor import CortexAInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert CortexAInferenceAdvisor.name() == "cortex_a_inference_advisor"


def test_configure_and_get_cortex_a_advisor(test_tflite_model: Path) -> None:
    """Test Cortex-A advisor configuration."""
    ctx = ExecutionContext()

    advisor = configure_and_get_cortexa_advisor(ctx, "cortex-a", test_tflite_model)
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, CortexAInferenceAdvisor)

    assert ctx.event_handlers is not None
    assert ctx.config_parameters == {
        "cortex_a_inference_advisor": {
            "model": str(test_tflite_model),
            "target_profile": "cortex-a",
        },
        "common_optimizations": {
            "optimizations": [_DEFAULT_OPTIMIZATION_TARGETS],
            "rewrite_parameters": {
                "train_params": None,
                "rewrite_specific_params": None,
            },
        },
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)


@pytest.mark.parametrize(
    "category, expected_error",
    [
        [
            AdviceCategory.PERFORMANCE,
            "Performance estimation is currently not supported for Cortex-A.",
        ]
    ],
)
def test_unsupported_advice_categories(
    tmp_path: Path,
    category: AdviceCategory,
    expected_error: str,
    test_tflite_model: Path,
) -> None:
    """Test that advisor should throw an exception for unsupported categories."""
    with pytest.raises(Exception, match=expected_error):
        ctx = ExecutionContext(output_dir=tmp_path, advice_category={category})

        advisor = configure_and_get_cortexa_advisor(ctx, "cortex-a", test_tflite_model)
        advisor.configure(ctx)
