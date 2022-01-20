# Copyright 2022, Arm Ltd.
"""Module for various helper classes."""
# pylint: disable=no-self-use, unused-argument
from typing import Any
from typing import List


class ActionResolver:
    """Helper class for generating actions (e.g. commands with parameters)."""

    def apply_optimizations(self, **kwargs: Any) -> List[str]:
        """Return action details for applying optimizations."""
        return []

    def supported_operators_info(self) -> List[str]:
        """Return action details for generating supported ops report."""
        return []

    def check_performance(self) -> List[str]:
        """Return action details for checking performance."""
        return []

    def check_operator_compatibility(self) -> List[str]:
        """Return action details for checking op compatibility."""
        return []

    def operator_compatibility_details(self) -> List[str]:
        """Return action details for getting more information about op compatibility."""
        return []

    def optimization_details(self) -> List[str]:
        """Return action detail for getting information about optimizations."""
        return []


class APIActionResolver(ActionResolver):
    """Helper class for the actions performed through API."""
