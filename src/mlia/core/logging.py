# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

from mlia.core.typing import OutputFormat
from mlia.utils.logging import attach_handlers
from mlia.utils.logging import create_log_handler
from mlia.utils.logging import NoASCIIFormatter


_CONSOLE_DEBUG_FORMAT = "%(name)s - %(levelname)s - %(message)s"
_FILE_DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    logs_dir: str | Path | None = None,
    verbose: bool = False,
    output_format: OutputFormat = "plain_text",
    log_filename: str = "mlia.log",
) -> None:
    """Set up logging.

    MLIA uses module 'logging' when it needs to produce output.

    :param logs_dir: path to the directory where application will save logs with
           debug information. If the path is not provided then no log files will
           be created during execution
    :param verbose: enable extended logging for the tools loggers
    :param output_format: specify the out format needed for setting up the right
           logging system
    :param log_filename: name of the log file in the logs directory
    """
    mlia_logger, tensorflow_logger, py_warnings_logger = (
        logging.getLogger(logger_name)
        for logger_name in ["mlia", "tensorflow", "py.warnings"]
    )

    # enable debug output, actual message filtering depends on
    # the provided parameters and being done at the handlers level
    for logger in [mlia_logger, tensorflow_logger]:
        logger.setLevel(logging.DEBUG)

    mlia_handlers = _get_mlia_handlers(logs_dir, log_filename, verbose, output_format)
    attach_handlers(mlia_handlers, [mlia_logger])

    tools_handlers = _get_tools_handlers(logs_dir, log_filename, verbose)
    attach_handlers(tools_handlers, [tensorflow_logger, py_warnings_logger])


def _get_mlia_handlers(
    logs_dir: str | Path | None,
    log_filename: str,
    verbose: bool,
    output_format: OutputFormat,
) -> Iterable[logging.Handler]:
    """Get handlers for the MLIA loggers."""
    # MLIA needs output to standard output via the logging system only when the
    # format is plain text. When the user specifies the "json" output format,
    # MLIA disables completely the logging system for the console output and it
    # relies on the print() function. This is needed because the output might
    # be corrupted with spurious messages in the standard output.
    if output_format == "plain_text":
        if verbose:
            log_level = logging.DEBUG
            log_format = _CONSOLE_DEBUG_FORMAT
        else:
            log_level = logging.INFO
            log_format = None

        # Create log handler for stdout
        yield create_log_handler(
            stream=sys.stdout, log_level=log_level, log_format=log_format
        )
    else:
        # In case of non plain text output, we need to inform the user if an
        # error happens during execution.
        yield create_log_handler(
            stream=sys.stderr,
            log_level=logging.ERROR,
        )

    # If the logs directory is specified, MLIA stores all output (according to
    # the logging level) into the file and removing the colouring of the
    # console output.
    if logs_dir:
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        yield create_log_handler(
            file_path=_get_log_file(logs_dir, log_filename),
            log_level=log_level,
            log_format=NoASCIIFormatter(fmt=_FILE_DEBUG_FORMAT),
            delay=True,
        )


def _get_tools_handlers(
    logs_dir: str | Path | None,
    log_filename: str,
    verbose: bool,
) -> Iterable[logging.Handler]:
    """Get handler for the tools loggers."""
    if verbose:
        yield create_log_handler(
            stream=sys.stdout,
            log_level=logging.DEBUG,
            log_format=_CONSOLE_DEBUG_FORMAT,
        )

    if logs_dir:
        yield create_log_handler(
            file_path=_get_log_file(logs_dir, log_filename),
            log_level=logging.DEBUG,
            log_format=_FILE_DEBUG_FORMAT,
            delay=True,
        )


def _get_log_file(logs_dir: str | Path, log_filename: str) -> Path:
    """Get the log file path."""
    logs_dir_path = Path(logs_dir)
    logs_dir_path.mkdir(exist_ok=True)

    return logs_dir_path / log_filename
