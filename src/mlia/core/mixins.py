# Copyright 2021, Arm Ltd.
"""Mixins module."""
from typing import Any
from typing import Optional

from mlia.core.context import Context


class ContextMixin:
    """Mixin for injecting context object."""

    context: Context

    def set_context(self, context: Context) -> None:
        """Context setter."""
        self.context = context


class ParameterResolverMixin:
    """Mixin for parameter resolving."""

    context: Context

    def get_parameter(
        self,
        name: str,
        expected: bool = True,
        expected_type: Optional[type] = None,
        context: Optional[Context] = None,
    ) -> Any:
        """Get parameter value."""
        ctx = context or self.context
        value = ctx.config_parameters.get(name)

        if not value and expected:
            raise Exception(f"Parameter {name} is not set")

        if value and expected_type is not None and not isinstance(value, expected_type):
            raise Exception(f"Parameter {name} expected to have type {expected_type}")

        return value

    def set_parameter(
        self, name: str, value: Any, context: Optional[Context] = None
    ) -> None:
        """Set parameter value."""
        ctx = context or self.context
        ctx.config_parameters[name] = value
