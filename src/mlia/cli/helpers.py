# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for various helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

from mlia.utils.boundaries.filesystem import copy_user_output_file


def copy_profile_file_to_output_dir(
    target_profile: str | Path, output_dir: str | Path, profile_to_copy: str
) -> bool:
    """Copy the target profile file to the output directory."""
    get_func_name = "get_builtin_" + profile_to_copy + "_path"
    get_func = getattr(importlib.import_module("mlia.target.config"), get_func_name)
    is_func_name = "is_builtin_" + profile_to_copy
    is_func = getattr(importlib.import_module("mlia.target.config"), is_func_name)
    profile_file_path = (
        get_func(cast(str, target_profile))
        if is_func(target_profile)
        else Path(target_profile)
    )
    output_file_path = f"{output_dir}/{profile_file_path.stem}.toml"
    try:
        copy_user_output_file(profile_file_path, output_file_path)
        return True
    except OSError as err:
        raise RuntimeError(f"Failed to copy {profile_to_copy} file: {err}") from err
