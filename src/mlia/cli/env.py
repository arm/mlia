# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI environment variable helpers."""

from __future__ import annotations

import os

from dotenv import dotenv_values


def get_environment() -> dict[str, str | None]:
    """Load the environment variables in place."""
    return {**dotenv_values(".env"), **os.environ}
