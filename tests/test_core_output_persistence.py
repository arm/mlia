# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical standardized-output persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.output_persistence import (
    STANDARDIZED_OUTPUT_FILENAME,
    persist_standardized_output,
)
from mlia.core.output_rendering import standardized_output_to_json


def test_persist_standardized_output_uses_stable_filename(tmp_path: Path) -> None:
    """Canonical output should use one stable filename in the output directory."""
    output: dict[str, object] = {
        "schema_version": "1.2.0",
        "results": [{"kind": "performance", "status": "ok"}],
    }

    output_path = persist_standardized_output(output, tmp_path)

    assert output_path == tmp_path / STANDARDIZED_OUTPUT_FILENAME
    assert output_path.name == "mlia-output.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == output


def test_persist_standardized_output_uses_user_output_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical output should be written through the user-output gateway."""
    write_user_output_text = MagicMock()
    monkeypatch.setattr(
        "mlia.core.output_persistence.write_user_output_text",
        write_user_output_text,
    )
    output = {"schema_version": "1.2.0", "results": []}

    output_path = persist_standardized_output(output, tmp_path)

    assert output_path == tmp_path / STANDARDIZED_OUTPUT_FILENAME
    write_user_output_text.assert_called_once_with(
        output_path,
        standardized_output_to_json(output) + "\n",
    )
