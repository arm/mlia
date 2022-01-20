# Copyright 2021, Arm Ltd.
"""Operators module."""
import logging

from mlia.tools import vela_wrapper


logger = logging.getLogger(__name__)


def generate_supported_operators_report() -> None:
    """Generate supported operators report."""
    vela_wrapper.generate_supported_operators_report()
