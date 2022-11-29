# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for various helper classes."""
# pylint: disable=unused-argument
from __future__ import annotations

from typing import Any


class ActionResolver:
    """Helper class for generating actions (e.g. commands with parameters)."""

    def apply_optimizations(self, **kwargs: Any) -> list[str]:
        """Return action details for applying optimizations."""
        return []

    def check_performance(self) -> list[str]:
        """Return action details for checking performance."""
        return []

    def check_operator_compatibility(self) -> list[str]:
        """Return action details for checking op compatibility."""
        return []

    def operator_compatibility_details(self) -> list[str]:
        """Return action details for getting more information about op compatibility."""
        return []

    def optimization_details(self) -> list[str]:
        """Return action detail for getting information about optimizations."""
        return []


class APIActionResolver(ActionResolver):
    """Helper class for the actions performed through API."""
