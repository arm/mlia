# Copyright (C) 2021-2022, Arm Ltd.
"""Tests for module options."""
from contextlib import ExitStack as does_not_raise
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pytest
from mlia.cli.options import get_target_opts
from mlia.cli.options import parse_optimization_parameters


@pytest.mark.parametrize(
    "optimization_type, optimization_target, expected_error, expected_result",
    [
        (
            "pruning",
            "0.5",
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                )
            ],
        ),
        (
            "clustering",
            "32",
            does_not_raise(),
            [
                dict(
                    optimization_type="clustering",
                    optimization_target=32.0,
                    layers_to_optimize=None,
                )
            ],
        ),
        (
            "pruning,clustering",
            "0.5,32",
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                dict(
                    optimization_type="clustering",
                    optimization_target=32.0,
                    layers_to_optimize=None,
                ),
            ],
        ),
        (
            "pruning, clustering",
            "0.5, 32",
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                dict(
                    optimization_type="clustering",
                    optimization_target=32.0,
                    layers_to_optimize=None,
                ),
            ],
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
        (
            "pruning,",
            "0.5,abc",
            pytest.raises(
                Exception, match="Non numeric value for the optimization target"
            ),
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


@pytest.mark.parametrize(
    "args, expected_opts",
    [
        [
            {},
            [],
        ],
        [
            {"target": "profile"},
            ["--target", "profile"],
        ],
        [
            # for the default profile empty list should be returned
            {"target": "U55-256"},
            [],
        ],
    ],
)
def test_get_target_opts(args: Optional[Dict], expected_opts: List[str]) -> None:
    """Test getting target options."""
    assert get_target_opts(args) == expected_opts
