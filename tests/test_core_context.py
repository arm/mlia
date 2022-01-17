# Copyright 2021, Arm Ltd.
"""Tests for the module context."""
from pathlib import Path

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.events import DefaultEventPublisher


def test_execution_context(tmpdir: str) -> None:
    """Test execution context."""
    publisher = DefaultEventPublisher()
    category = AdviceCategory.OPERATORS_COMPATIBILITY

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
        f"working_dir={Path(tmpdir)}, "
        "advice_category=OPERATORS_COMPATIBILITY, "
        "config_parameters={'param': 'value'}, "
        "verbose=True"
    )

    context_with_default_params = ExecutionContext()
    assert context_with_default_params.advice_category is None
    assert context_with_default_params.config_parameters is None
    assert context_with_default_params.event_handlers == []
    assert isinstance(
        context_with_default_params.event_publisher, DefaultEventPublisher
    )
    assert context_with_default_params.logs_path == Path.cwd() / "logs"

    default_model_path = context_with_default_params.get_model_path("sample.model")
    expected_default_model_path = Path.cwd() / "models/sample.model"
    assert default_model_path == expected_default_model_path

    expected_str = (
        f"ExecutionContext: working_dir={Path.cwd()}, "
        "advice_category=<not set>, "
        "config_parameters=None, "
        "verbose=False"
    )
    assert str(context_with_default_params) == expected_str
