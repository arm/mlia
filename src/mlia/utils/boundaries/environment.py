# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Boundaries for temporary environment configuration."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import dotenv_values


@contextmanager
def temporary_dotenv_environment() -> Iterator[None]:
    """Apply dotenv defaults for a scope and then restore the environment."""
    original_environment = os.environ.copy()
    for name, value in dotenv_values().items():
        if value is not None:
            os.environ.setdefault(name, value)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
