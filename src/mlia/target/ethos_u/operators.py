# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Operators module."""
import logging

from mlia.backend.vela.compat import generate_supported_operators_report


logger = logging.getLogger(__name__)


def report() -> None:
    """Generate supported operators report."""
    generate_supported_operators_report()
