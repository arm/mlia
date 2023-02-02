# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module options."""
from __future__ import annotations

import argparse
from contextlib import ExitStack as does_not_raise
from typing import Any

import pytest

from mlia.cli.options import get_output_format
from mlia.cli.options import get_target_profile_opts
from mlia.cli.options import parse_optimization_parameters
from mlia.core.typing import OutputFormat


@pytest.mark.parametrize(
    "pruning, clustering, pruning_target, clustering_target, expected_error,"
    "expected_result",
    [
        [
            False,
            False,
            None,
            None,
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                )
            ],
        ],
        [
            True,
            False,
            None,
            None,
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                )
            ],
        ],
        [
            False,
            True,
            None,
            None,
            does_not_raise(),
            [
                dict(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                )
            ],
        ],
        [
            True,
            True,
            None,
            None,
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                dict(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                ),
            ],
        ],
        [
            False,
            False,
            0.4,
            None,
            does_not_raise(),
            [
                dict(
                    optimization_type="pruning",
                    optimization_target=0.4,
                    layers_to_optimize=None,
                )
            ],
        ],
        [
            False,
            False,
            None,
            32,
            pytest.raises(
                argparse.ArgumentError,
                match="To enable clustering optimization you need to include "
                "the `--clustering` flag in your command.",
            ),
            None,
        ],
        [
            False,
            True,
            None,
            32.2,
            does_not_raise(),
            [
                dict(
                    optimization_type="clustering",
                    optimization_target=32.2,
                    layers_to_optimize=None,
                )
            ],
        ],
    ],
)
def test_parse_optimization_parameters(
    pruning: bool,
    clustering: bool,
    pruning_target: float | None,
    clustering_target: int | None,
    expected_error: Any,
    expected_result: Any,
) -> None:
    """Test function parse_optimization_parameters."""
    with expected_error:
        result = parse_optimization_parameters(
            pruning, clustering, pruning_target, clustering_target
        )
        assert result == expected_result


@pytest.mark.parametrize(
    "args, expected_opts",
    [
        [
            {},
            [],
        ],
        [
            {"target_profile": "profile"},
            ["--target-profile", "profile"],
        ],
        [
            # for the default profile empty list should be returned
            {"target": "ethos-u55-256"},
            [],
        ],
    ],
)
def test_get_target_opts(args: dict | None, expected_opts: list[str]) -> None:
    """Test getting target options."""
    assert get_target_profile_opts(args) == expected_opts


@pytest.mark.parametrize(
    "args, expected_output_format",
    [
        [
            {},
            "plain_text",
        ],
        [
            {"json": True},
            "json",
        ],
        [
            {"json": False},
            "plain_text",
        ],
    ],
)
def test_get_output_format(args: dict, expected_output_format: OutputFormat) -> None:
    """Test get_output_format function."""
    arguments = argparse.Namespace(**args)
    output_format = get_output_format(arguments)
    assert output_format == expected_output_format
