# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A data collection module."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.context import ExecutionContext
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.data_collection import CortexAOperatorCompatibility
from mlia.target.cortex_a.operators import CortexACompatibilityInfo

CORTEX_A_CONFIG = CortexAConfiguration.load_profile("cortex-a")
VERSION = CORTEX_A_CONFIG.armnn_tflite_delegate_version


def check_cortex_a_data_collection(
    monkeypatch: pytest.MonkeyPatch, model: Path, tmpdir: str
) -> None:
    """Test Cortex-A data collection."""
    assert CortexAOperatorCompatibility.name()

    monkeypatch.setattr(
        "mlia.target.cortex_a.data_collection.get_cortex_a_compatibility_info",
        MagicMock(return_value=CortexACompatibilityInfo([], VERSION)),
    )

    context = ExecutionContext(output_dir=tmpdir)
    collector = CortexAOperatorCompatibility(model, CORTEX_A_CONFIG)
    collector.set_context(context)

    data_item = collector.collect_data()

    assert isinstance(data_item, CortexACompatibilityInfo)


def test_cortex_a_data_collection_tflite(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path, tmpdir: str
) -> None:
    """Test Cortex-A data collection with a TensorFlow Lite model."""
    check_cortex_a_data_collection(monkeypatch, test_tflite_model, tmpdir)


def test_cortex_a_data_collection_keras(
    monkeypatch: pytest.MonkeyPatch, test_keras_model: Path, tmpdir: str
) -> None:
    """Test Cortex-A data collection with a Keras model."""
    check_cortex_a_data_collection(monkeypatch, test_keras_model, tmpdir)


def test_cortex_a_data_collection_tf(
    monkeypatch: pytest.MonkeyPatch, test_tf_model: Path, tmpdir: str
) -> None:
    """Test Cortex-A data collection with a SavedModel."""
    check_cortex_a_data_collection(monkeypatch, test_tf_model, tmpdir)


def test_cortex_a_data_collection_incompatible(tmpdir: str) -> None:
    """Test Cortex-A data collection with an incompatible model."""
    context = ExecutionContext(output_dir=tmpdir)
    collector = CortexAOperatorCompatibility(
        "model.notflite", CORTEX_A_CONFIG  # type: ignore
    )
    collector.set_context(context)

    data_item = collector.collect_data()

    assert isinstance(data_item, TFLiteCompatibilityInfo)
    assert data_item.status != TFLiteCompatibilityStatus.COMPATIBLE
