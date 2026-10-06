# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified logging boundary gateways."""

from __future__ import annotations

import logging
from pathlib import Path


def create_file_handler(
    file_path: str | Path, *, delay: bool = True
) -> logging.FileHandler:
    """Create a handler for logs written to a user-requested file."""
    return logging.FileHandler(file_path, delay=delay)
