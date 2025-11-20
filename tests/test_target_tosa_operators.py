# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tosa-checker operators."""
import pytest

from mlia.target.tosa.operators import report


def test_tosa_operators_report() -> None:
    """Test operators report function."""
    with pytest.raises(
        NotImplementedError,
        match="Generating a supported operators report is not "
        "currently supported with TOSA target profile.",
    ):
        report()
