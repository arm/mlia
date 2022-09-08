# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
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
        [["--output", "report.csv"], "report.csv"],
        [["--output", "REPORT.CSV"], "REPORT.CSV"],
        [["--output", "some_folder/report.csv"], "some_folder/report.csv"],
    ],
)
def test_output_options(output_parameters: list[str], expected_path: str) -> None:
    """Test output options resolving."""
    parser = argparse.ArgumentParser()
    add_output_options(parser)

    args = parser.parse_args(output_parameters)
    assert args.output == expected_path


@pytest.mark.parametrize(
    "output_filename",
    [
        "report.txt",
        "report.TXT",
        "report",
        "report.pdf",
    ],
)
def test_output_options_bad_parameters(
    output_filename: str, capsys: pytest.CaptureFixture
) -> None:
    """Test that args parsing should fail if format is not supported."""
    parser = argparse.ArgumentParser()
    add_output_options(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--output", output_filename])

    err_output = capsys.readouterr().err
    suffix = Path(output_filename).suffix[1:]
    assert f"Unsupported format '{suffix}'" in err_output
