# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Various util functions."""


def yes(prompt: str) -> bool:
    """Return true if user confirms the action."""
    response = input(f"{prompt} [y/n]: ")
    return response in ["y", "Y"]
