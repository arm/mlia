# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reports module."""
from __future__ import annotations

from mlia.core.advice_generation import Advice
from mlia.core.reporting import Column
from mlia.core.reporting import Report
from mlia.core.reporting import Table


def report_advice(advice: list[Advice]) -> Report:
    """Generate report for the advice."""
    return Table(
        columns=[
            Column("#", only_for=["plain_text"]),
            Column("Advice", alias="advice_message"),
        ],
        rows=[(i + 1, a.messages) for i, a in enumerate(advice)],
        name="Advice",
        alias="advice",
    )
