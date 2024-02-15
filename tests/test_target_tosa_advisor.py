# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA advisor."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.tosa.advisor import configure_and_get_tosa_advisor
from mlia.target.tosa.advisor import TOSAInferenceAdvisor


def test_configure_and_get_tosa_advisor(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test TOSA advisor configuration."""
    ctx = ExecutionContext()
    get_events_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.target.tosa.advisor.TOSAInferenceAdvisor.get_events",
        MagicMock(return_value=get_events_mock),
    )

    advisor = configure_and_get_tosa_advisor(ctx, "tosa", test_tflite_model)
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, TOSAInferenceAdvisor)

    assert advisor.get_events(ctx) == get_events_mock
    assert ctx.event_handlers is not None
    assert ctx.config_parameters == {
        "common_optimizations": {
            "optimizations": [
                [
                    {
                        "layers_to_optimize": None,
                        "optimization_target": 0.5,
                        "optimization_type": "pruning",
                    },
                    {
                        "layers_to_optimize": None,
                        "optimization_target": 32,
                        "optimization_type": "clustering",
                    },
                ]
            ],
            "training_parameters": [None],
        },
        "tosa_inference_advisor": {
            "model": str(test_tflite_model),
            "target_profile": "tosa",
        },
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)


@pytest.mark.parametrize(
    "category, expected_error",
    [
        [
            AdviceCategory.PERFORMANCE,
            "Performance estimation is currently not supported for TOSA.",
        ],
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

        advisor = configure_and_get_tosa_advisor(ctx, "tosa", test_tflite_model)
        advisor.configure(ctx)
