# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Persistence for canonical standardized MLIA output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlia.core.output_rendering import standardized_output_to_json

STANDARDIZED_OUTPUT_FILENAME = "mlia-output.json"


def persist_standardized_output(output: dict[str, Any], output_dir: Path) -> Path:
    """Write canonical standardized output and return its stable path."""
    output_path = output_dir / STANDARDIZED_OUTPUT_FILENAME
    output_path.write_text(
        standardized_output_to_json(output) + "\n",
        encoding="utf-8",
    )
    return output_path
