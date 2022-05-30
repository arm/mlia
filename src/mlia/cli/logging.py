# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI logging configuration."""
import logging
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from mlia.utils.logging import attach_handlers
from mlia.utils.logging import create_log_handler
from mlia.utils.logging import LogFilter


_CONSOLE_DEBUG_FORMAT = "%(name)s - %(message)s"
_FILE_DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    logs_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    log_filename: str = "mlia.log",
) -> None:
    """Set up logging.

    MLIA uses module 'logging' when it needs to produce output.

    :param logs_dir: path to the directory where application will save logs with
           debug information. If the path is not provided then no log files will
           be created during execution
    :param verbose: enable extended logging for the tools loggers
    :param log_filename: name of the log file in the logs directory
    """
    mlia_logger, *tools_loggers = [
        logging.getLogger(logger_name)
        for logger_name in ["mlia", "tensorflow", "py.warnings"]
    ]

    # enable debug output, actual message filtering depends on
    # the provided parameters and being done on the handlers level
    mlia_logger.setLevel(logging.DEBUG)

    mlia_handlers = _get_mlia_handlers(logs_dir, log_filename, verbose)
    attach_handlers(mlia_handlers, [mlia_logger])

    tools_handlers = _get_tools_handlers(logs_dir, log_filename, verbose)
    attach_handlers(tools_handlers, tools_loggers)


def _get_mlia_handlers(
    logs_dir: Optional[Union[str, Path]],
    log_filename: str,
    verbose: bool,
) -> List[logging.Handler]:
    """Get handlers for the MLIA loggers."""
    result = []
    stdout_handler = create_log_handler(
        stream=sys.stdout,
        log_level=logging.INFO,
    )
    result.append(stdout_handler)

    if verbose:
        mlia_verbose_handler = create_log_handler(
            stream=sys.stdout,
            log_level=logging.DEBUG,
            log_format=_CONSOLE_DEBUG_FORMAT,
            log_filter=LogFilter.equals(logging.DEBUG),
        )
        result.append(mlia_verbose_handler)

    if logs_dir:
        mlia_file_handler = create_log_handler(
            file_path=_get_log_file(logs_dir, log_filename),
            log_level=logging.DEBUG,
            log_format=_FILE_DEBUG_FORMAT,
            log_filter=LogFilter.skip(logging.INFO),
            delay=True,
        )
        result.append(mlia_file_handler)

    return result


def _get_tools_handlers(
    logs_dir: Optional[Union[str, Path]],
    log_filename: str,
    verbose: bool,
) -> List[logging.Handler]:
    """Get handler for the tools loggers."""
    result = []
    if verbose:
        verbose_stdout_handler = create_log_handler(
            stream=sys.stdout,
            log_level=logging.DEBUG,
            log_format=_CONSOLE_DEBUG_FORMAT,
        )
        result.append(verbose_stdout_handler)

    if logs_dir:
        file_handler = create_log_handler(
            file_path=_get_log_file(logs_dir, log_filename),
            log_level=logging.DEBUG,
            log_format=_FILE_DEBUG_FORMAT,
            delay=True,
        )
        result.append(file_handler)

    return result


def _get_log_file(logs_dir: Union[str, Path], log_filename: str) -> Path:
    """Get the log file path."""
    logs_dir_path = Path(logs_dir)
    logs_dir_path.mkdir(exist_ok=True)
    return logs_dir_path / log_filename
