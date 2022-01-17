# Copyright 2021, Arm Ltd.
"""Tests for module advisor."""
from unittest.mock import MagicMock

from mlia.core.advisor import InferenceAdvisor
from mlia.core.context import Context
from mlia.core.workflow import WorkflowExecutor


def test_inference_advisor_run() -> None:
    """Test running sample inference advisor."""
    executor_mock = MagicMock(spec=WorkflowExecutor)
    context_mock = MagicMock(spec=Context)

    class SampleAdvisor(InferenceAdvisor):
        """Sample inference advisor."""

        @classmethod
        def name(cls) -> str:
            """Return name of the advisor."""
            return "sample_advisor"

        @classmethod
        def description(cls) -> str:
            """Return description of the advisor."""
            return "Sample advisor"

        @classmethod
        def info(cls) -> None:
            """Print advisor info."""

        def configure(self, context: Context) -> WorkflowExecutor:
            """Configure advisor."""
            return executor_mock

    advisor = SampleAdvisor()
    advisor.run(context_mock)

    executor_mock.run.assert_called_once()
