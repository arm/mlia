# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module cli.logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mlia.core.logging import setup_logging
from mlia.core.typing import OutputFormat
from mlia.utils.logging import log_boundary_action
from tests.utils.logging import clear_loggers


def teardown_function() -> None:
    """Perform action after test completion.

    This function is launched automatically by pytest after each test
    in this module.
    """
    clear_loggers()


@pytest.mark.parametrize(
    "logs_dir,verbose,output_format,expected_output,expected_log_file_content",
    [
        (
            None,
            False,
            "plain_text",
            """cli info
cli error
""",
            None,
        ),
        (
            None,
            True,
            "plain_text",
            """mlia.backend.manager - DEBUG - backends debug
mlia.cli - INFO - cli info
mlia.cli - DEBUG - cli debug
mlia.cli - ERROR - cli error
""",
            None,
        ),
        (
            "logs",
            True,
            "plain_text",
            """mlia.backend.manager - DEBUG - backends debug
mlia.cli - INFO - cli info
mlia.cli - DEBUG - cli debug
mlia.cli - ERROR - cli error
""",
            """mlia.backend.manager - DEBUG - backends debug
mlia.cli - INFO - cli info
mlia.cli - DEBUG - cli debug
mlia.cli - ERROR - cli error
""",
        ),
        (
            "logs",
            False,
            "json",
            "",
            """mlia.cli - INFO - cli info
mlia.cli - ERROR - cli error
""",
        ),
    ],
)
def test_setup_logging(
    tmp_path: Path,
    capfd: pytest.CaptureFixture,
    logs_dir: str | None,
    verbose: bool,
    output_format: OutputFormat,
    expected_output: str,
    expected_log_file_content: str,
) -> None:
    """Test function setup_logging."""
    logs_dir_path = tmp_path / logs_dir if logs_dir else None

    setup_logging(logs_dir_path, verbose, output_format, cli_mode=True)

    backend_logger = logging.getLogger("mlia.backend.manager")
    backend_logger.debug("backends debug")

    cli_logger = logging.getLogger("mlia.cli")
    cli_logger.info("cli info")
    cli_logger.debug("cli debug")
    cli_logger.error("cli error")

    stdout, _ = capfd.readouterr()
    assert stdout == expected_output

    check_log_assertions(logs_dir_path, expected_log_file_content)


def test_setup_logging_defaults_to_cli_console_output(
    capfd: pytest.CaptureFixture,
) -> None:
    """Existing callers retain console logging without passing CLI mode."""
    setup_logging()

    logging.getLogger("mlia.test").info("application info")

    stdout, stderr = capfd.readouterr()
    assert stdout == "application info\n"
    assert stderr == ""


def test_setup_logging_defaults_to_cli_verbose_tool_output(
    capfd: pytest.CaptureFixture,
) -> None:
    """Existing verbose callers retain tool console logging."""
    setup_logging(verbose=True)

    logging.getLogger("tensorflow").debug("tensorflow detail")

    stdout, stderr = capfd.readouterr()
    assert "tensorflow detail" in stdout
    assert stderr == ""


def test_setup_logging_closes_replaced_file_handlers(tmp_path: Path) -> None:
    """Reconfiguring logging releases files owned by the previous setup."""
    first_logs = tmp_path / "first"
    second_logs = tmp_path / "second"

    setup_logging(first_logs, verbose=False, output_format="json")
    first_file = first_logs / "mlia.log"
    assert first_file.is_file()

    setup_logging(second_logs, verbose=False, output_format="json")

    first_file.unlink()
    assert not first_file.exists()


def test_json_logging_sends_only_boundary_info_and_errors_to_stderr(
    capfd: pytest.CaptureFixture,
) -> None:
    """Boundary notices remain visible without corrupting JSON stdout."""
    setup_logging(output_format="json", cli_mode=True)
    logger = logging.getLogger("mlia.test")

    logger.info("ordinary info")
    log_boundary_action(logger, "runtime boundary")
    logger.error("failure")

    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert "ordinary info" not in stderr
    assert "runtime boundary" in stderr
    assert "failure" in stderr


def test_non_cli_json_logging_writes_only_to_file(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """Library JSON logging does not install a console handler."""
    logs_dir = tmp_path / "logs"
    setup_logging(logs_dir, output_format="json", cli_mode=False)
    logger = logging.getLogger("mlia.test")

    log_boundary_action(logger, "runtime boundary")
    logger.error("failure")

    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stderr == ""
    content = (logs_dir / "mlia.log").read_text(encoding="utf-8")
    assert "runtime boundary" in content
    assert "failure" in content


def test_non_cli_verbose_tool_logging_writes_only_to_file(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """Library calls do not install tool console handlers in verbose mode."""
    logs_dir = tmp_path / "logs"
    setup_logging(logs_dir, verbose=True, output_format="json", cli_mode=False)

    logging.getLogger("tensorflow").debug("tensorflow detail")
    logging.getLogger("py.warnings").warning("library warning")

    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stderr == ""
    content = (logs_dir / "mlia.log").read_text(encoding="utf-8")
    assert "tensorflow detail" in content
    assert "library warning" in content


def test_cli_verbose_tool_logging_writes_to_stdout(
    capfd: pytest.CaptureFixture,
) -> None:
    """Verbose tool output remains visible when invoked through the CLI."""
    setup_logging(verbose=True, output_format="plain_text", cli_mode=True)

    logging.getLogger("tensorflow").debug("tensorflow detail")

    stdout, stderr = capfd.readouterr()
    assert "tensorflow detail" in stdout
    assert stderr == ""


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
