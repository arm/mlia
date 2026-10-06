# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified temporary-resource boundary gateways."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from typing import Generator, TextIO


@contextmanager
def temporary_text_stream() -> Generator[TextIO, None, None]:
    """Create a seekable temporary text stream."""
    with tempfile.TemporaryFile(mode="r+") as stream:
        yield stream
