# Copyright 2021, Arm Ltd.
"""Tests for module options."""
from contextlib import ExitStack as does_not_raise
from typing import Any

import pytest
from mlia.cli.options import parse_optimization_parameters


@pytest.mark.parametrize(
    "optimization_type, optimization_target, expected_error, expected_result",
    [
        (
            "pruning",
            "0.5",
            does_not_raise(),
            [("pruning", 0.5)],
        ),
        (
            "clustering",
            "32",
            does_not_raise(),
            [("clustering", 32.0)],
        ),
        (
            "pruning,clustering",
            "0.5,32",
            does_not_raise(),
            [("pruning", 0.5), ("clustering", 32.0)],
        ),
        (
            "pruning, clustering",
            "0.5, 32",
            does_not_raise(),
            [("pruning", 0.5), ("clustering", 32.0)],
        ),
        (
            "pruning,clustering",
            "0.5",
            pytest.raises(
                Exception, match="Wrong number of optimization targets and types"
            ),
            None,
        ),
        (
            "",
            "0.5",
            pytest.raises(Exception, match="Optimization type is not provided"),
            None,
        ),
        (
            "pruning,clustering",
            "",
            pytest.raises(Exception, match="Optimization target is not provided"),
            None,
        ),
    ],
)
def test_parse_optimization_parameters(
    optimization_type: str,
    optimization_target: str,
    expected_error: Any,
    expected_result: Any,
) -> None:
    """Test function parse_optimization_parameters."""
    with expected_error:
        result = parse_optimization_parameters(optimization_type, optimization_target)
        assert result == expected_result
