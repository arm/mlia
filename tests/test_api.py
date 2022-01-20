# Copyright 2022, Arm Ltd.
"""Tests for the API functions."""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mlia.api import get_advice
from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.context import ExecutionContext


def test_get_advice_no_target_provided(test_models_path: Path) -> None:
    """Test getting advice when no target provided."""
    model = test_models_path / "simple_model.h5"

    with pytest.raises(Exception, match="Target is not provided"):
        get_advice(None, model, "all")  # type: ignore


def test_get_advice_wrong_category(test_models_path: Path) -> None:
    """Test getting advice when wrong advice category provided."""
    model = test_models_path / "simple_model.h5"

    with pytest.raises(Exception, match="Invalid advice category unknown"):
        get_advice("U55-256", model, "unknown")  # type: ignore


@pytest.mark.parametrize(
    "category, context, expected_category",
    [
        [
            "all",
            None,
            AdviceCategory.ALL,
        ],
        [
            "optimization",
            None,
            AdviceCategory.OPTIMIZATION,
        ],
        [
            "operators",
            None,
            AdviceCategory.OPERATORS,
        ],
        [
            "performance",
            None,
            AdviceCategory.PERFORMANCE,
        ],
        [
            "all",
            ExecutionContext(),
            AdviceCategory.ALL,
        ],
        [
            "all",
            ExecutionContext(advice_category=AdviceCategory.PERFORMANCE),
            AdviceCategory.PERFORMANCE,
        ],
        [
            "all",
            ExecutionContext(config_parameters={"param": "value"}),
            AdviceCategory.ALL,
        ],
        [
            "all",
            ExecutionContext(event_handlers=[MagicMock()]),
            AdviceCategory.ALL,
        ],
    ],
)
def test_get_advice(
    test_models_path: Path,
    monkeypatch: Any,
    category: str,
    context: ExecutionContext,
    expected_category: AdviceCategory,
) -> None:
    """Test getting advice with valid parameters."""
    advisor_mock = MagicMock()
    monkeypatch.setattr("mlia.api._get_advisor", MagicMock(return_value=advisor_mock))

    model = test_models_path / "simple_model.h5"
    get_advice("U55-256", model, category, context=context)  # type: ignore

    advisor_mock.run.assert_called_once()
    context = advisor_mock.run.mock_calls[0].args[0]
    assert isinstance(context, Context)
    assert context.advice_category == expected_category

    assert context.event_handlers is not None
    assert context.config_parameters is not None
