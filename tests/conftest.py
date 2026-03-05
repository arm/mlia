# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest conf module for core tests."""

from pathlib import Path

import pytest

from mlia.core.context import ExecutionContext


@pytest.fixture(name="sample_context")
def fixture_sample_context(tmpdir: str) -> ExecutionContext:
    """Return sample context fixture."""
    return ExecutionContext(output_dir=tmpdir)


@pytest.fixture(scope="session", name="test_models_path")
def fixture_test_models_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide path to the test models."""
    tmp_path = tmp_path_factory.mktemp("models")
    (tmp_path / "model.tflite").write_text("test", encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="session", name="test_tflite_model")
def fixture_test_tflite_model(test_models_path: Path) -> Path:
    """Return test TFLite model."""
    return test_models_path / "model.tflite"
