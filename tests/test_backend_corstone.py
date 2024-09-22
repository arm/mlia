# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Corstone backend."""
from mlia.backend.corstone import is_corstone_backend


def test_is_corstone_backend() -> None:
    """Test function is_corstone_backend."""
    assert is_corstone_backend("corstone-300") is True
    assert is_corstone_backend("corstone-310") is True
    assert is_corstone_backend("New backend") is False
