# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the API functions."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.api import get_advice
from mlia.api import get_advisor
from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.devices.ethosu.advisor import EthosUInferenceAdvisor
from mlia.devices.tosa.advisor import TOSAInferenceAdvisor


def test_get_advice_no_target_provided(test_keras_model: Path) -> None:
    """Test getting advice when no target provided."""
    with pytest.raises(Exception, match="Target profile is not provided"):
        get_advice(None, test_keras_model, "all")  # type: ignore


def test_get_advice_wrong_category(test_keras_model: Path) -> None:
    """Test getting advice when wrong advice category provided."""
    with pytest.raises(Exception, match="Invalid advice category unknown"):
        get_advice("ethos-u55-256", test_keras_model, "unknown")  # type: ignore


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
            AdviceCategory.ALL,
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
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    context: ExecutionContext,
    expected_category: AdviceCategory,
    test_keras_model: Path,
) -> None:
    """Test getting advice with valid parameters."""
    advisor_mock = MagicMock()
    monkeypatch.setattr("mlia.api.get_advisor", MagicMock(return_value=advisor_mock))

    get_advice(
        "ethos-u55-256",
        test_keras_model,
        category,  # type: ignore
        context=context,
    )

    advisor_mock.run.assert_called_once()
    context = advisor_mock.run.mock_calls[0].args[0]
    assert isinstance(context, Context)
    assert context.advice_category == expected_category


def test_get_advisor(
    test_keras_model: Path,
) -> None:
    """Test function for getting the advisor."""
    ethos_u55_advisor = get_advisor(
        ExecutionContext(), "ethos-u55-256", str(test_keras_model)
    )
    assert isinstance(ethos_u55_advisor, EthosUInferenceAdvisor)

    tosa_advisor = get_advisor(ExecutionContext(), "tosa", str(test_keras_model))
    assert isinstance(tosa_advisor, TOSAInferenceAdvisor)
