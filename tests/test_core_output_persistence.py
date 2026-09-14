# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical standardized-output persistence."""

from __future__ import annotations

import json
from pathlib import Path

from mlia.core.output_persistence import (
    STANDARDIZED_OUTPUT_FILENAME,
    persist_standardized_output,
)


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
