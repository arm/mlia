# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module options."""
from __future__ import annotations

import argparse
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest

from mlia.cli.options import add_output_options
from mlia.cli.options import get_target_profile_opts
from mlia.cli.options import parse_optimization_parameters
from mlia.cli.options import parse_output_parameters
from mlia.core.common import FormattedFilePath


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
    "output_parameters,  expected_path",
    [
        [["--output", "report.json"], "report.json"],
        [["--output", "REPORT.JSON"], "REPORT.JSON"],
        [["--output", "some_folder/report.json"], "some_folder/report.json"],
    ],
)
def test_output_options(output_parameters: list[str], expected_path: str) -> None:
    """Test output options resolving."""
    parser = argparse.ArgumentParser()
    add_output_options(parser)

    args = parser.parse_args(output_parameters)
    assert str(args.output) == expected_path


@pytest.mark.parametrize(
    "path, json, expected_error, output",
    [
        [
            None,
            True,
            pytest.raises(
                argparse.ArgumentError,
                match=r"To enable JSON output you need to specify the output path. "
                r"\(e.g. --output out.json --json\)",
            ),
            None,
        ],
        [None, False, does_not_raise(), None],
        [
            Path("test_path"),
            False,
            does_not_raise(),
            FormattedFilePath(Path("test_path"), "plain_text"),
        ],
        [
            Path("test_path"),
            True,
            does_not_raise(),
            FormattedFilePath(Path("test_path"), "json"),
        ],
    ],
)
def test_parse_output_parameters(
    path: Path | None, json: bool, expected_error: Any, output: FormattedFilePath | None
) -> None:
    """Test parsing for output parameters."""
    with expected_error:
        formatted_output = parse_output_parameters(path, json)
        assert formatted_output == output
