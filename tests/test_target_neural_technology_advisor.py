# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology MLIA module."""
from pathlib import Path

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.neural_technology.advisor import (
    configure_and_get_neural_technology_advisor,
)
from mlia.target.neural_technology.advisor import NeuralTechnologyInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert (
        NeuralTechnologyInferenceAdvisor.name() == "neural_technology_inference_advisor"
    )


def test_configure_and_get_neural_technology_advisor(test_tflite_model: Path) -> None:
    """Test Neural Technology advisor configuration."""
    ctx = ExecutionContext(advice_category={AdviceCategory.PERFORMANCE})

    advisor = configure_and_get_neural_technology_advisor(
        ctx, "neural-technology", test_tflite_model, backends=["nx-graph-compiler"]
    )
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, NeuralTechnologyInferenceAdvisor)

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
            "rewrite_parameters": {
                "rewrite_specific_params": None,
                "train_params": None,
            },
        },
        "neural_technology_inference_advisor": {
            "backends": ["nx-graph-compiler"],
            "model": str(test_tflite_model),
            "target_profile": "neural-technology",
        },
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)
