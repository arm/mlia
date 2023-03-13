# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Hydra MLIA module."""
from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.hydra.advisor import configure_and_get_hydra_advisor
from mlia.target.hydra.advisor import HydraInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert HydraInferenceAdvisor.name() == "hydra_inference_advisor"


def test_configure_and_get_hydra_advisor(test_tflite_model: Path) -> None:
    """Test Hydra advisor configuration."""
    ctx = ExecutionContext(advice_category={AdviceCategory.PERFORMANCE})

    advisor = configure_and_get_hydra_advisor(ctx, "hydra", test_tflite_model)
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, HydraInferenceAdvisor)

    assert ctx.event_handlers is not None
    assert ctx.config_parameters == {
        "hydra_inference_advisor": {
            "model": str(test_tflite_model),
            "target_profile": "hydra",
        }
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)


@pytest.mark.parametrize(
    "category", (AdviceCategory.COMPATIBILITY, AdviceCategory.OPTIMIZATION)
)
def test_unsupported_advice_categories(
    tmp_path: Path,
    category: AdviceCategory,
    test_tflite_model: Path,
) -> None:
    """Test that advisor should throw an exception for unsupported categories."""
    with pytest.raises(ValueError):
        ctx = ExecutionContext(output_dir=tmp_path, advice_category={category})
        advisor = configure_and_get_hydra_advisor(ctx, "hydra", test_tflite_model)
        advisor.configure(ctx)
