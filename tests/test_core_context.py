# Copyright 2021, Arm Ltd.
"""Tests for the module context."""
from pathlib import Path

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.events import DefaultEventPublisher


def test_execution_context(tmpdir: str) -> None:
    """Test execution context."""
    publisher = DefaultEventPublisher()

    context = ExecutionContext(
        advice_categories=[
            AdviceCategory.OPERATORS_COMPATIBILITY,
            AdviceCategory.OPTIMIZATION,
        ],
        config_parameters={"param": "value"},
        working_dir=tmpdir,
        event_handlers=[],
        event_publisher=publisher,
        verbose=True,
        logs_dir="logs_directory",
        models_dir="models_directory",
    )

    assert context.advice_categories == [
        AdviceCategory.OPERATORS_COMPATIBILITY,
        AdviceCategory.OPTIMIZATION,
    ]
    assert context.config_parameters == {"param": "value"}
    assert context.event_handlers == []
    assert context.event_publisher == publisher
    assert context.logs_path == Path(tmpdir) / "logs_directory"
    expected_model_path = Path(tmpdir) / "models_directory/sample.model"
    assert context.get_model_path("sample.model") == expected_model_path
    assert context.verbose is True
    assert (
        str(context) == f"ExecutionContext: "
        f"working_dir={Path(tmpdir)}, "
        "advice_categories=[OPERATORS_COMPATIBILITY,OPTIMIZATION], "
        "config_parameters={'param': 'value'}, "
        "verbose=True"
    )
