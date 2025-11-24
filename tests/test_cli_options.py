# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module options."""
from __future__ import annotations

import argparse
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mlia.cli.options import add_backend_install_options
from mlia.cli.options import add_backend_options
from mlia.cli.options import add_backend_uninstall_options
from mlia.cli.options import add_check_category_options
from mlia.cli.options import add_dataset_options
from mlia.cli.options import add_debug_options
from mlia.cli.options import add_keras_model_options
from mlia.cli.options import add_model_options
from mlia.cli.options import add_multi_optimization_options
from mlia.cli.options import add_output_directory
from mlia.cli.options import add_output_options
from mlia.cli.options import add_target_options
from mlia.cli.options import get_output_format
from mlia.cli.options import get_target_profile_opts
from mlia.cli.options import parse_optimization_parameters
from mlia.core.common import AdviceCategory
from mlia.core.errors import ConfigurationError
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
                {
                    "optimization_type": "pruning",
                    "optimization_target": 0.5,
                    "layers_to_optimize": None,
                    "dataset": None,
                }
            ],
        ],
        [
            True,
            False,
            None,
            None,
            does_not_raise(),
            [
                {
                    "optimization_type": "pruning",
                    "optimization_target": 0.5,
                    "layers_to_optimize": None,
                    "dataset": None,
                }
            ],
        ],
        [
            False,
            True,
            None,
            None,
            does_not_raise(),
            [
                {
                    "optimization_type": "clustering",
                    "optimization_target": 32,
                    "layers_to_optimize": None,
                    "dataset": None,
                }
            ],
        ],
        [
            True,
            True,
            None,
            None,
            does_not_raise(),
            [
                {
                    "optimization_type": "pruning",
                    "optimization_target": 0.5,
                    "layers_to_optimize": None,
                    "dataset": None,
                },
                {
                    "optimization_type": "clustering",
                    "optimization_target": 32,
                    "layers_to_optimize": None,
                    "dataset": None,
                },
            ],
        ],
        [
            False,
            False,
            0.4,
            None,
            does_not_raise(),
            [
                {
                    "optimization_type": "pruning",
                    "optimization_target": 0.4,
                    "layers_to_optimize": None,
                    "dataset": None,
                }
            ],
        ],
        [
            False,
            False,
            None,
            32,
            pytest.raises(argparse.ArgumentError),
            None,
        ],
        [
            False,
            True,
            None,
            32.2,
            does_not_raise(),
            [
                {
                    "optimization_type": "clustering",
                    "optimization_target": 32.2,
                    "layers_to_optimize": None,
                    "dataset": None,
                }
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
        [
            # Test list handling in construct_param
            {"target_profile": ["profile1", "profile2"]},
            ["--target-profile", "profile1", "--target-profile", "profile2"],
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


@pytest.mark.parametrize(
    "rewrite, rewrite_target, rewrite_start, rewrite_end, expected_error",
    [
        [
            True,
            None,
            "start_node",
            "end_node",
            pytest.raises(ConfigurationError),
        ],
        [
            True,
            "some_target",
            None,
            "end_node",
            pytest.raises(ConfigurationError),
        ],
        [
            True,
            "some_target",
            "start_node",
            None,
            pytest.raises(ConfigurationError),
        ],
    ],
)
def test_parse_optimization_parameters_rewrite_missing_params(
    rewrite: bool,
    rewrite_target: str | None,
    rewrite_start: str | None,
    rewrite_end: str | None,
    expected_error: Any,
) -> None:
    """Test parse_optimization_parameters raises error when rewrite params missing."""
    with expected_error:
        parse_optimization_parameters(
            rewrite=rewrite,
            rewrite_target=rewrite_target,
            rewrite_start=rewrite_start,
            rewrite_end=rewrite_end,
        )


def test_parse_optimization_parameters_rewrite_invalid_target() -> None:
    """Test parse_optimization_parameters raises error for invalid rewrite target."""
    with patch("mlia.cli.options.RewritingOptimizer.builtin_rewrite_names") as mock:
        mock.return_value = ["valid_rewrite_1", "valid_rewrite_2"]

        with pytest.raises(
            ConfigurationError,
            match="Invalid rewrite target: 'invalid_rewrite'",
        ):
            parse_optimization_parameters(
                rewrite=True,
                rewrite_target="invalid_rewrite",
                rewrite_start="start",
                rewrite_end="end",
            )


def test_parse_optimization_parameters_rewrite_valid() -> None:
    """Test parse_optimization_parameters with valid rewrite parameters."""
    with patch("mlia.cli.options.RewritingOptimizer.builtin_rewrite_names") as mock:
        mock.return_value = ["valid_rewrite"]

        result = parse_optimization_parameters(
            rewrite=True,
            rewrite_target="valid_rewrite",
            rewrite_start="start_node",
            rewrite_end="end_node",
        )

        assert len(result) == 1
        assert result[0]["optimization_type"] == "rewrite"
        assert result[0]["optimization_target"] == "valid_rewrite"
        assert result[0]["layers_to_optimize"] == ["start_node", "end_node"]
        assert result[0]["dataset"] is None


def test_parse_optimization_parameters_with_dataset_and_layers() -> None:
    """Test parse_optimization_parameters with dataset and layers_to_optimize."""
    dataset_path = Path("/path/to/dataset.tfrec")
    layers = ["layer1", "layer2"]

    result = parse_optimization_parameters(
        pruning=True,
        pruning_target=0.7,
        dataset=dataset_path,
        layers_to_optimize=layers,
    )

    assert len(result) == 1
    assert result[0]["optimization_type"] == "pruning"
    assert result[0]["optimization_target"] == pytest.approx(0.7)
    assert result[0]["layers_to_optimize"] == layers
    assert result[0]["dataset"] == dataset_path


def test_add_check_category_options() -> None:
    """Test add_check_category_options adds correct arguments."""
    parser = argparse.ArgumentParser()
    add_check_category_options(parser)

    args = parser.parse_args(["--performance", "--compatibility"])
    assert args.performance is True
    assert args.compatibility is True

    args = parser.parse_args([])
    assert args.performance is False
    assert args.compatibility is False


def test_add_target_options() -> None:
    """Test add_target_options adds target-profile argument."""
    parser = argparse.ArgumentParser()
    add_target_options(parser, required=False)

    args = parser.parse_args(["--target-profile", "ethos-u55-256"])
    assert args.target_profile == "ethos-u55-256"

    args = parser.parse_args(["-t", "cortex-a"])
    assert args.target_profile == "cortex-a"


@pytest.mark.parametrize(
    "supported_advice",
    [
        [AdviceCategory.PERFORMANCE, AdviceCategory.COMPATIBILITY],
        [AdviceCategory.OPTIMIZATION],
    ],
    ids=["performance_and_compatibility", "optimization"],
)
def test_add_target_options_with_supported_advice(
    supported_advice: list[AdviceCategory],
) -> None:
    """Test add_target_options filters profiles based on supported advice."""
    parser = argparse.ArgumentParser()
    add_target_options(parser, supported_advice=supported_advice, required=False)

    # Should accept target profiles that support the specified advice categories
    args = parser.parse_args(["--target-profile", "ethos-u55-256"])
    assert args.target_profile == "ethos-u55-256"


def test_add_multi_optimization_options() -> None:
    """Test add_multi_optimization_options adds all optimization arguments."""
    parser = argparse.ArgumentParser()
    add_multi_optimization_options(parser)

    args = parser.parse_args(
        [
            "--pruning",
            "--clustering",
            "--rewrite",
            "--pruning-target",
            "0.6",
            "--clustering-target",
            "16",
            "--rewrite-target",
            "conv2d",
            "--rewrite-start",
            "node1",
            "--rewrite-end",
            "node2",
            "--optimization-profile",
            "custom",
        ]
    )

    assert args.pruning is True
    assert args.clustering is True
    assert args.rewrite is True
    assert args.pruning_target == pytest.approx(0.6)
    assert args.clustering_target == 16
    assert args.rewrite_target == "conv2d"
    assert args.rewrite_start == "node1"
    assert args.rewrite_end == "node2"
    assert args.optimization_profile == "custom"


def test_add_model_options() -> None:
    """Test add_model_options adds model argument."""
    parser = argparse.ArgumentParser()
    add_model_options(parser)

    args = parser.parse_args(["model.tflite"])
    assert args.model == "model.tflite"


def test_add_output_options() -> None:
    """Test add_output_options adds json flag."""
    parser = argparse.ArgumentParser()
    add_output_options(parser)

    args = parser.parse_args(["--json"])
    assert args.json is True

    args = parser.parse_args([])
    assert args.json is False


def test_add_debug_options() -> None:
    """Test add_debug_options adds debug flag."""
    parser = argparse.ArgumentParser()
    add_debug_options(parser)

    args = parser.parse_args(["--debug"])
    assert args.debug is True

    args = parser.parse_args(["-d"])
    assert args.debug is True

    args = parser.parse_args([])
    assert args.debug is False


def test_add_dataset_options() -> None:
    """Test add_dataset_options adds dataset argument."""
    parser = argparse.ArgumentParser()
    add_dataset_options(parser)

    args = parser.parse_args(["--dataset", "/path/to/data.tfrec"])
    assert args.dataset == Path("/path/to/data.tfrec")


def test_add_keras_model_options() -> None:
    """Test add_keras_model_options adds model argument."""
    parser = argparse.ArgumentParser()
    add_keras_model_options(parser)

    args = parser.parse_args(["model.h5"])
    assert args.model == "model.h5"


def test_add_backend_install_options(tmp_path: Path) -> None:
    """Test add_backend_install_options adds all install arguments."""
    parser = argparse.ArgumentParser()
    add_backend_install_options(parser)

    # Test with valid directory
    valid_dir = tmp_path / "install"
    valid_dir.mkdir()

    args = parser.parse_args(
        [
            "--path",
            str(valid_dir),
            "--i-agree-to-the-contained-eula",
            "--force",
            "--noninteractive",
            "backend1",
            "backend2",
        ]
    )

    assert args.path == valid_dir
    assert args.i_agree_to_the_contained_eula is True
    assert args.force is True
    assert args.noninteractive is True
    assert args.names == ["backend1", "backend2"]


def test_add_backend_install_options_invalid_directory(tmp_path: Path) -> None:
    """Test add_backend_install_options rejects invalid directory."""
    parser = argparse.ArgumentParser()
    add_backend_install_options(parser)

    invalid_path = tmp_path / "does_not_exist"

    with pytest.raises(SystemExit):
        parser.parse_args(["--path", str(invalid_path), "backend"])


def test_add_backend_uninstall_options() -> None:
    """Test add_backend_uninstall_options adds names argument."""
    parser = argparse.ArgumentParser()
    add_backend_uninstall_options(parser)

    args = parser.parse_args(["backend1"])
    assert args.names == ["backend1"]

    args = parser.parse_args(["backend1", "backend2", "backend3"])
    assert args.names == ["backend1", "backend2", "backend3"]


def test_add_output_directory() -> None:
    """Test add_output_directory adds output-dir argument."""
    parser = argparse.ArgumentParser()
    add_output_directory(parser)

    args = parser.parse_args(["--output-dir", "/path/to/output"])
    assert args.output_dir == Path("/path/to/output")


def test_add_backend_options() -> None:
    """Test add_backend_options adds backend argument."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.get_available_backends") as mock_backends:
        mock_backends.return_value = ["vela", "tosa-checker", "corstone-300"]
        add_backend_options(parser)

        args = parser.parse_args(["-b", "vela"])
        assert args.backend == ["vela"]

        args = parser.parse_args(["--backend", "vela", "--backend", "tosa-checker"])
        assert args.backend == ["vela", "tosa-checker"]


def test_add_backend_options_multiple_corstone_error() -> None:
    """Test add_backend_options rejects multiple Corstone backends."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.get_available_backends") as mock_backends:
        with patch("mlia.cli.options.is_corstone_backend") as mock_is_corstone:
            mock_backends.return_value = ["corstone-300", "corstone-310"]
            mock_is_corstone.return_value = True

            add_backend_options(parser)

            with pytest.raises(SystemExit):
                parser.parse_args(
                    ["--backend", "corstone-300", "--backend", "corstone-310"]
                )


def test_add_backend_options_with_skip_list() -> None:
    """Test add_backend_options with backends_to_skip."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.get_available_backends") as mock_backends:
        mock_backends.return_value = ["vela", "tosa-checker", "armnn-tflite-delegate"]
        add_backend_options(parser, backends_to_skip=["tosa-checker"])

        # tosa-checker should not be in choices
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "tosa-checker"])
