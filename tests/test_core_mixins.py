# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the module mixins."""
import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.mixins import ContextMixin
from mlia.core.mixins import ParameterResolverMixin


def test_context_mixin(sample_context: Context) -> None:
    """Test ContextMixin."""

    class SampleClass(ContextMixin):
        """Sample class."""

    sample_object = SampleClass()
    sample_object.set_context(sample_context)
    assert sample_object.context == sample_context


class TestParameterResolverMixin:
    """Tests for parameter resolver mixin."""

    @staticmethod
    def test_parameter_resolver_mixin(sample_context: ExecutionContext) -> None:
        """Test ParameterResolverMixin."""

        class SampleClass(ParameterResolverMixin):
            """Sample class."""

            def __init__(self) -> None:
                """Init sample object."""
                self.context = sample_context

                self.context.update(
                    advice_category=AdviceCategory.OPERATORS,
                    event_handlers=[],
                    config_parameters={"section": {"param": 123}},
                )

        sample_object = SampleClass()
        value = sample_object.get_parameter("section", "param")
        assert value == 123

        with pytest.raises(
            Exception, match="Parameter param expected to have type <class 'str'>"
        ):
            value = sample_object.get_parameter("section", "param", expected_type=str)

        with pytest.raises(Exception, match="Parameter no_param is not set"):
            value = sample_object.get_parameter("section", "no_param")

    @staticmethod
    def test_parameter_resolver_mixin_no_config(
        sample_context: ExecutionContext,
    ) -> None:
        """Test ParameterResolverMixin without config params."""

        class SampleClassNoConfig(ParameterResolverMixin):
            """Sample context without config params."""

            def __init__(self) -> None:
                """Init sample object."""
                self.context = sample_context

        with pytest.raises(Exception, match="Configuration parameters are not set"):
            sample_object_no_config = SampleClassNoConfig()
            sample_object_no_config.get_parameter("section", "param", expected_type=str)

    @staticmethod
    def test_parameter_resolver_mixin_bad_section(
        sample_context: ExecutionContext,
    ) -> None:
        """Test ParameterResolverMixin without config params."""

        class SampleClassBadSection(ParameterResolverMixin):
            """Sample context with bad section in config."""

            def __init__(self) -> None:
                """Init sample object."""
                self.context = sample_context
                self.context.update(
                    advice_category=AdviceCategory.OPERATORS,
                    event_handlers=[],
                    config_parameters={"section": ["param"]},
                )

        with pytest.raises(
            Exception,
            match="Parameter section section has wrong format, "
            "expected to be a dictionary",
        ):
            sample_object_bad_section = SampleClassBadSection()
            sample_object_bad_section.get_parameter(
                "section", "param", expected_type=str
            )
