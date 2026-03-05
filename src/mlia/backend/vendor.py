# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for vendor artifacts."""

from __future__ import annotations

from pathlib import Path

import mlia


def vendor_artifact_path(backend_folder: str) -> Path | None:
    """Return mlia/_vendor/artifacts/<backend_folder> if it exists, else None."""
    for pkg_path in mlia.__path__:
        pkg_root = Path(pkg_path)
        vendor = pkg_root / "_vendor" / "artifacts" / backend_folder
        if vendor.exists():
            return vendor
    return None
