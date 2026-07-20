# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight shell-completion inventories."""

from __future__ import annotations

from pathlib import Path

import mlia


def target_profile_names() -> tuple[str, ...]:
    """Return packaged target profile names from MLIA namespace resources."""
    profile_names: set[str] = set()
    for namespace_dir in mlia.__path__:
        profiles_dir = Path(namespace_dir) / "resources" / "target_profiles"
        try:
            profile_names.update(
                profile.stem
                for profile in profiles_dir.iterdir()
                if profile.is_file() and profile.suffix == ".toml"
            )
        except OSError:
            continue

    return tuple(sorted(profile_names))
