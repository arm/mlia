# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module context."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.events import DefaultEventPublisher


@pytest.mark.parametrize(
    "context_advice_category, expected_enabled_categories",
    [
        [
            {
                AdviceCategory.COMPATIBILITY,
            },
            [AdviceCategory.COMPATIBILITY],
        ],
        [
            {
                AdviceCategory.PERFORMANCE,
            },
            [AdviceCategory.PERFORMANCE],
        ],
        [
            {AdviceCategory.COMPATIBILITY, AdviceCategory.PERFORMANCE},
            [AdviceCategory.PERFORMANCE, AdviceCategory.COMPATIBILITY],
        ],
    ],
)
def test_execution_context_category_enabled(
    context_advice_category: set[AdviceCategory],
    expected_enabled_categories: list[AdviceCategory],
) -> None:
    """Test category enabled method of execution context."""
    for category in expected_enabled_categories:
        assert ExecutionContext(
            advice_category=context_advice_category
        ).category_enabled(category)


def test_execution_context(tmpdir: str) -> None:
    """Test execution context."""
    publisher = DefaultEventPublisher()
    category = {AdviceCategory.COMPATIBILITY}

    context = ExecutionContext(
        advice_category=category,
        config_parameters={"param": "value"},
        working_dir=tmpdir,
        event_handlers=[],
        event_publisher=publisher,
        verbose=True,
        logs_dir="logs_directory",
        models_dir="models_directory",
    )

    assert context.advice_category == category
    assert context.config_parameters == {"param": "value"}
    assert context.event_handlers == []
    assert context.event_publisher == publisher
    assert context.logs_path == Path(tmpdir) / "logs_directory"
    expected_model_path = Path(tmpdir) / "models_directory/sample.model"
    assert context.get_model_path("sample.model") == expected_model_path
    assert context.verbose is True
    assert str(context) == (
        f"ExecutionContext: "
        f"working_dir={tmpdir}, "
        "advice_category={'COMPATIBILITY'}, "
        "config_parameters={'param': 'value'}, "
        "verbose=True"
    )

    context_with_default_params = ExecutionContext(working_dir=tmpdir)
    assert context_with_default_params.advice_category == {AdviceCategory.COMPATIBILITY}
    assert context_with_default_params.config_parameters is None
    assert context_with_default_params.event_handlers is None
    assert isinstance(
        context_with_default_params.event_publisher, DefaultEventPublisher
    )
    assert context_with_default_params.logs_path == Path(tmpdir) / "logs"

    default_model_path = context_with_default_params.get_model_path("sample.model")
    expected_default_model_path = Path(tmpdir) / "models/sample.model"
    assert default_model_path == expected_default_model_path

    expected_str = (
        f"ExecutionContext: working_dir={tmpdir}, "
        "advice_category={'COMPATIBILITY'}, "
        "config_parameters=None, "
        "verbose=False"
    )
    assert str(context_with_default_params) == expected_str
