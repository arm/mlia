# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module context."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.events import DefaultEventPublisher
from mlia.utils.filesystem import USER_ONLY_PERM_MASK
from mlia.utils.filesystem import working_directory
from tests.utils.common import check_expected_permissions


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
        ctx = ExecutionContext(advice_category=context_advice_category)
        assert ctx.category_enabled(category)


def test_execution_context(tmp_path: Path) -> None:
    """Test execution context."""
    publisher = DefaultEventPublisher()
    category = {AdviceCategory.COMPATIBILITY}

    context = ExecutionContext(
        advice_category=category,
        config_parameters={"param": "value"},
        output_dir=tmp_path / "output",
        event_handlers=[],
        event_publisher=publisher,
        verbose=True,
        logs_dir="logs_directory",
        models_dir="models_directory",
        output_format="json",
    )

    output_dir = context.output_dir
    assert output_dir == tmp_path.joinpath("output", "mlia-output")
    assert output_dir.is_dir()
    check_expected_permissions(output_dir, USER_ONLY_PERM_MASK)
    check_expected_permissions(tmp_path.joinpath("output"), USER_ONLY_PERM_MASK)

    assert context.advice_category == category
    assert context.config_parameters == {"param": "value"}
    assert context.event_handlers == []
    assert context.event_publisher == publisher
    assert context.logs_path == output_dir / "logs_directory"
    expected_model_path = output_dir / "models_directory/sample.model"
    assert context.get_model_path("sample.model") == expected_model_path
    assert context.verbose is True
    assert context.output_format == "json"
    assert str(context) == (
        f"ExecutionContext: "
        f"output_dir={output_dir}, "
        "advice_category={'COMPATIBILITY'}, "
        "config_parameters={'param': 'value'}, "
        "verbose=True, "
        "output_format=json"
    )


def test_execution_context_with_default_params(tmp_path: Path) -> None:
    """Test execution context with the default parameters."""
    working_dir = tmp_path / "sample"
    with working_directory(working_dir, create_dir=True):
        context_with_default_params = ExecutionContext()

    assert context_with_default_params.advice_category == {AdviceCategory.COMPATIBILITY}
    assert context_with_default_params.config_parameters is None
    assert context_with_default_params.event_handlers is None
    assert isinstance(
        context_with_default_params.event_publisher, DefaultEventPublisher
    )

    output_dir = context_with_default_params.output_dir
    assert output_dir == working_dir.joinpath("mlia-output")

    assert context_with_default_params.logs_path == output_dir / "logs"

    default_model_path = context_with_default_params.get_model_path("sample.model")
    expected_default_model_path = output_dir / "models/sample.model"
    assert default_model_path == expected_default_model_path
    assert context_with_default_params.output_format == "plain_text"

    expected_str = (
        f"ExecutionContext: output_dir={output_dir}, "
        "advice_category={'COMPATIBILITY'}, "
        "config_parameters=None, "
        "verbose=False, "
        "output_format=plain_text"
    )
    assert str(context_with_default_params) == expected_str
