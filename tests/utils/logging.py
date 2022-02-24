# Copyright (C) 2021-2022, Arm Ltd.
"""Utils for logging."""
import logging


def clear_loggers() -> None:
    """Close the log handlers."""
    for _, logger in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger, logging.PlaceHolder):
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)
