# Copyright 2021, Arm Ltd.
"""Tests for the module cli.logging."""
import logging
import sys
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Optional

import pytest
from mlia.cli.logging import create_log_handler
from mlia.cli.logging import setup_logging

from tests.utils.logging import clear_loggers


def teardown_function() -> None:
    """Perform action after test completion.

    This function is launched automatically by pytest after each test
    in this module.
    """
    clear_loggers()


@pytest.mark.parametrize(
    "file_path, stream, log_level, log_format, delay, "
    "expected_error, expected_class",
    [
        (
            "test.log",
            None,
            logging.INFO,
            "%(name)s - %(message)s",
            True,
            does_not_raise(),
            logging.FileHandler,
        ),
        (
            None,
            sys.stdout,
            logging.INFO,
            "%(name)s - %(message)s",
            None,
            does_not_raise(),
            logging.StreamHandler,
        ),
        (
            None,
            None,
            logging.INFO,
            "%(name)s - %(message)s",
            None,
            pytest.raises(Exception, match="Unable to create logging handler"),
            None,
        ),
    ],
)
def test_create_log_handler(
    file_path: Optional[Path],
    stream: Optional[Any],
    log_level: Optional[int],
    log_format: Optional[str],
    delay: bool,
    expected_error: Any,
    expected_class: type,
) -> None:
    """Test function test_create_log_handler."""
    with expected_error:
        handler = create_log_handler(
            file_path=file_path,
            stream=stream,
            log_level=log_level,
            log_format=log_format,
            delay=delay,
        )
        assert isinstance(handler, expected_class)


@pytest.mark.parametrize(
    "logs_dir, verbose, expected_output",
    [
        (
            None,
            None,
            "cli info\n",
        ),
        (
            None,
            True,
            "mlia.tools.aiet - aiet debug\ncli info\n",
        ),
        (
            "logs",
            True,
            "mlia.tools.aiet - aiet debug\ncli info\n",
        ),
    ],
)
def test_setup_logging(
    tmp_path: Path, capfd: Any, logs_dir: str, verbose: bool, expected_output: str
) -> None:
    """Test function setup_logging."""
    logs_dir_path = tmp_path / logs_dir if logs_dir else None

    setup_logging(str(logs_dir_path) if logs_dir_path is not None else None, verbose)

    aiet_logger = logging.getLogger("mlia.tools.aiet")
    aiet_logger.debug("aiet debug")

    cli_logger = logging.getLogger("mlia.cli")
    cli_logger.info("cli info")
    cli_logger.debug("cli debug")

    stdout, _ = capfd.readouterr()
    assert stdout == expected_output

    if logs_dir_path is not None:
        items = list(logs_dir_path.iterdir())
        assert len(items) == 1

        log_file_name = items[0].name
        assert log_file_name.endswith("aiet.log")
