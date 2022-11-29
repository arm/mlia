# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the helper classes."""
from mlia.core.helpers import APIActionResolver


def test_api_action_resolver() -> None:
    """Test APIActionResolver class."""
    helper = APIActionResolver()

    # pylint: disable=use-implicit-booleaness-not-comparison
    assert helper.apply_optimizations() == []
    assert helper.check_performance() == []
    assert helper.check_operator_compatibility() == []
    assert helper.operator_compatibility_details() == []
    assert helper.optimization_details() == []
