# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reports module."""
from __future__ import annotations

from typing import Any
from typing import Callable

from mlia.core.advice_generation import Advice
from mlia.core.reporting import Report
from mlia.devices.cortexa.config import CortexAConfiguration
from mlia.devices.cortexa.operators import Operator
from mlia.utils.types import is_list_of


def report_device(device: CortexAConfiguration) -> Report:
    """Generate report for the device."""
    raise NotImplementedError()


def report_advice(advice: list[Advice]) -> Report:
    """Generate report for the advice."""
    raise NotImplementedError()


def report_cortex_a_operators(operators: list[Operator]) -> Report:
    """Generate report for the operators."""
    raise NotImplementedError()


def cortex_a_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, CortexAConfiguration):
        return report_device

    if is_list_of(data, Operator):
        return report_cortex_a_operators

    raise Exception(f"Unable to find appropriate formatter for {data}")
