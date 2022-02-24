# Copyright (C) 2021-2022, Arm Ltd.
"""Tests for the helper classes."""
from mlia.core.helpers import APIActionResolver


def test_api_action_resolver() -> None:
    """Test APIActionResolver class."""
    helper = APIActionResolver()

    assert helper.apply_optimizations() == []
    assert helper.supported_operators_info() == []
    assert helper.check_performance() == []
    assert helper.check_operator_compatibility() == []
    assert helper.operator_compatibility_details() == []
    assert helper.optimization_details() == []
