# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common test utils module."""

from __future__ import annotations

from pathlib import Path


def check_expected_permissions(path: Path, expected_permissions_mask: int) -> None:
    """Check expected permissions for the provided path."""
    path_mode = path.stat().st_mode
    permissions_mask = path_mode & 0o777

    assert permissions_mask == expected_permissions_mask
