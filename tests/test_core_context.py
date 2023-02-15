# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module context."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import create_output_dir_with_timestamp
from mlia.core.context import ExecutionContext
from mlia.core.events import DefaultEventPublisher
from mlia.utils.filesystem import working_directory


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


def test_execution_context(tmpdir: str) -> None:
    """Test execution context."""
    publisher = DefaultEventPublisher()
    category = {AdviceCategory.COMPATIBILITY}

    context = ExecutionContext(
        advice_category=category,
        config_parameters={"param": "value"},
        output_dir=tmpdir,
        event_handlers=[],
        event_publisher=publisher,
        verbose=True,
        logs_dir="logs_directory",
        models_dir="models_directory",
        output_format="json",
    )

    assert context.advice_category == category
    assert context.config_parameters == {"param": "value"}
    assert context.event_handlers == []
    assert context.event_publisher == publisher
    assert context.logs_path == Path(tmpdir) / "logs_directory"
    expected_model_path = Path(tmpdir) / "models_directory/sample.model"
    assert context.get_model_path("sample.model") == expected_model_path
    assert context.verbose is True
    assert context.output_format == "json"
    assert str(context) == (
        f"ExecutionContext: "
        f"output_dir={tmpdir}, "
        "advice_category={'COMPATIBILITY'}, "
        "config_parameters={'param': 'value'}, "
        "verbose=True, "
        "output_format=json"
    )


def test_execution_context_with_default_params(tmpdir: str) -> None:
    """Test execution context with the default parameters."""
    context_with_default_params = ExecutionContext(output_dir=tmpdir)
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
    assert context_with_default_params.output_format == "plain_text"

    expected_str = (
        f"ExecutionContext: output_dir={tmpdir}, "
        "advice_category={'COMPATIBILITY'}, "
        "config_parameters=None, "
        "verbose=False, "
        "output_format=plain_text"
    )
    assert str(context_with_default_params) == expected_str


def test_create_output_dir_with_timestamp(tmp_path: Path) -> None:
    """Test function generate_output_dir_with_timestamp."""
    working_dir = tmp_path / "sample"
    with working_directory(working_dir, create_dir=True):
        create_output_dir_with_timestamp()

    working_dir_content = list(working_dir.iterdir())
    assert len(working_dir_content) == 1

    parent_dir = working_dir_content[0]
    assert parent_dir.is_dir()
    assert parent_dir.name == "mlia-output"

    parent_dir_content = list(parent_dir.iterdir())
    assert len(parent_dir_content) == 1

    output_dir = parent_dir_content[0]
    assert output_dir.is_dir()

    pattern = re.compile(r"mlia-output-\d{4}-\d{2}-\d{2}T\d+[.]\d+")
    assert pattern.match(output_dir.name)

    output_dir_content = list(output_dir.iterdir())
    assert not output_dir_content
