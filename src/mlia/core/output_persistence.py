# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Persistence for canonical standardized MLIA output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlia.core.output_rendering import standardized_output_to_json
from mlia.utils.boundaries.filesystem import write_user_output_text

STANDARDIZED_OUTPUT_FILENAME = "mlia-output.json"


def persist_standardized_output(output: dict[str, Any], output_dir: Path) -> Path:
    """Write canonical standardized output and return its stable path."""
    output_path = output_dir / STANDARDIZED_OUTPUT_FILENAME
    write_user_output_text(output_path, standardized_output_to_json(output) + "\n")
    return output_path
