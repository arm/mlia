# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Helpers functions."""
import logging
from typing import Any


def set_verbosity(
    ctx: Any, option: Any, verbosity: Any  # pylint: disable=unused-argument
) -> None:
    """Set the logging level according to the verbosity."""
    # Unused arguments must be present here in definition as these are required in
    # function definition when set as a callback
    if verbosity == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif verbosity > 1:
        logging.getLogger().setLevel(logging.DEBUG)
