# Copyright 2022, Arm Ltd.
"""Tests for the logging utility functions."""
import logging
import sys
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Optional

import pytest
from mlia.cli.logging import create_log_handler


@pytest.mark.parametrize(
    "file_path, stream, log_filter, delay, expected_error, expected_class",
    [
        (
            "test.log",
            None,
            None,
            True,
            does_not_raise(),
            logging.FileHandler,
        ),
        (
            None,
            sys.stdout,
            None,
            None,
            does_not_raise(),
            logging.StreamHandler,
        ),
        (
            None,
            None,
            None,
            None,
            pytest.raises(Exception, match="Unable to create logging handler"),
            None,
        ),
    ],
)
def test_create_log_handler(
    file_path: Optional[Path],
    stream: Optional[Any],
    log_filter: Optional[logging.Filter],
    delay: bool,
    expected_error: Any,
    expected_class: type,
) -> None:
    """Test function test_create_log_handler."""
    with expected_error:
        handler = create_log_handler(
            file_path=file_path,
            stream=stream,
            log_level=logging.INFO,
            log_format="%(name)s - %(message)s",
            log_filter=log_filter,
            delay=delay,
        )
        assert isinstance(handler, expected_class)
