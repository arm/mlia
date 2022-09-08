# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module cli.logging."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mlia.cli.logging import setup_logging
from tests.utils.logging import clear_loggers


def teardown_function() -> None:
    """Perform action after test completion.

    This function is launched automatically by pytest after each test
    in this module.
    """
    clear_loggers()


@pytest.mark.parametrize(
    "logs_dir, verbose, expected_output, expected_log_file_content",
    [
        (
            None,
            None,
            "cli info\n",
            None,
        ),
        (
            None,
            True,
            """mlia.backend.manager - backends debug
cli info
mlia.cli - cli debug
""",
            None,
        ),
        (
            "logs",
            True,
            """mlia.backend.manager - backends debug
cli info
mlia.cli - cli debug
""",
            """mlia.backend.manager - DEBUG - backends debug
mlia.cli - DEBUG - cli debug
""",
        ),
    ],
)
def test_setup_logging(
    tmp_path: Path,
    capfd: pytest.CaptureFixture,
    logs_dir: str,
    verbose: bool,
    expected_output: str,
    expected_log_file_content: str,
) -> None:
    """Test function setup_logging."""
    logs_dir_path = tmp_path / logs_dir if logs_dir else None

    setup_logging(logs_dir_path, verbose)

    backend_logger = logging.getLogger("mlia.backend.manager")
    backend_logger.debug("backends debug")

    cli_logger = logging.getLogger("mlia.cli")
    cli_logger.info("cli info")
    cli_logger.debug("cli debug")

    stdout, _ = capfd.readouterr()
    assert stdout == expected_output

    check_log_assertions(logs_dir_path, expected_log_file_content)


def check_log_assertions(
    logs_dir_path: Path | None, expected_log_file_content: str
) -> None:
    """Test assertions for log file."""
    if logs_dir_path is not None:
        assert logs_dir_path.is_dir()

        items = list(logs_dir_path.iterdir())
        assert len(items) == 1

        log_file_path = items[0]
        assert log_file_path.is_file()

        log_file_name = log_file_path.name
        assert log_file_name == "mlia.log"

        with open(log_file_path, encoding="utf-8") as log_file:
            log_content = log_file.read()

        expected_lines = expected_log_file_content.split("\n")
        produced_lines = log_content.split("\n")

        assert len(expected_lines) == len(produced_lines)
        for expected, produced in zip(expected_lines, produced_lines):
            assert expected in produced
