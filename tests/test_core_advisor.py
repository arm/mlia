# SPDX-FileCopyrightText: Copyright 2022, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module advisor."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.advisor import DefaultInferenceAdvisor, InferenceAdvisor
from mlia.core.context import Context
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.settings import ApplicationSettings, FilteringSettings
from mlia.core.workflow import DefaultWorkflowExecutor, WorkflowExecutor


def test_inference_advisor_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The advisor should postprocess its authoritative output exactly once."""
    raw_output: dict[str, object] = {"results": []}
    processed_output: dict[str, object] = {"processed": True}
    executor_mock = MagicMock(spec=WorkflowExecutor)
    executor_mock.run.return_value = raw_output
    postprocess = MagicMock(return_value=processed_output)
    monkeypatch.setattr(
        "mlia.core.advisor.postprocess_standardized_output", postprocess
    )
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
    settings = ApplicationSettings(filtering=FilteringSettings(collapse=()))

    assert advisor.run(context_mock, settings) == processed_output
    executor_mock.run.assert_called_once_with()
    postprocess.assert_called_once_with(raw_output, settings)


def test_default_inference_advisor(test_tflite_model: Path) -> None:
    """Test DefaultInferenceAdvisor abstract class."""
    advisor_name = "my_default_advisor"
    target_profile = "some_profile"
    context_mock = MagicMock(spec=Context)
    context_mock.config_parameters = {
        advisor_name: {
            "param": "value",
            "model": test_tflite_model.as_posix(),
            "target_profile": target_profile,
        }
    }

    data_collector_mock = MagicMock(spec=DataCollector)
    data_analyzer_mock = MagicMock(spec=DataAnalyzer)

    class MyDefaultInferenceAdvisor(DefaultInferenceAdvisor):
        """Sample DefaultInferenceAdvisor."""

        @classmethod
        def name(cls) -> str:
            """Return name of the advisor."""
            return advisor_name

        def get_collectors(self, context: Context) -> list[DataCollector]:
            """Return list of the data collectors."""
            return [data_collector_mock]

        def get_analyzers(self, context: Context) -> list[DataAnalyzer]:
            """Return list of the data analyzers."""
            return [data_analyzer_mock]

        def get_pattern_analyzers(self, _context: Context) -> list:
            """Return list of the pattern analyzers."""
            return []

    advisor = MyDefaultInferenceAdvisor()
    workflow_executor = advisor.configure(context_mock)
    assert isinstance(workflow_executor, DefaultWorkflowExecutor)
    assert workflow_executor.context == context_mock
    assert workflow_executor.collectors == [data_collector_mock]
    assert workflow_executor.analyzers == [data_analyzer_mock]

    assert advisor.get_string_parameter(context_mock, "param") == "value"
    assert advisor.get_model(context_mock) == test_tflite_model
    assert advisor.get_target_profile(context_mock) == target_profile

    bad_model_context_mock = MagicMock(spec=Context)
    bad_model_context_mock.config_parameters = {
        advisor_name: {"param": "value", "model": "no_such_model.tflite"}
    }
    with pytest.raises(FileNotFoundError, match="does not exist."):
        _ = advisor.get_model(bad_model_context_mock)
