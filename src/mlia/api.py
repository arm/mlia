# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the API functions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.registry import get_optimization_profile
from mlia.target.registry import profile
from mlia.target.registry import registry as target_registry

logger = logging.getLogger(__name__)


def get_advice(
    target_profile: str,
    model: str | Path,
    category: set[str],
    optimization_profile: str | None = None,
    optimization_targets: list[dict[str, Any]] | None = None,
    context: ExecutionContext | None = None,
    backends: list[str] | None = None,
) -> None:
    """Get the advice.

    This function represents an entry point to the library API.

    Based on provided parameters it will collect and analyze the data
    and produce the advice.

    :param target_profile: target profile identifier
    :param model: path to the NN model
    :param category: set of categories of the advice. MLIA supports three categories:
           "compatibility", "performance", "optimization". If not provided
           category "compatibility" is used by default.
    :param optimization_targets: optional model optimization targets that
           could be used for generating advice in "optimization" category.
    :param context: optional parameter which represents execution context,
           could be used for advanced use cases
    :param backends: A list of backends that should be used for the given
           target. Default settings will be used if None.

    Examples:
        NB: Before launching MLIA, the logging functionality should be configured!

        Getting the advice for the provided target profile and the model

        >>> get_advice("ethos-u55-256", "path/to/the/model",
                       {"optimization", "compatibility"})

        Getting the advice for the category "performance".

        >>> get_advice("ethos-u55-256", "path/to/the/model", {"performance"})

    """
    advice_category = AdviceCategory.from_string(category)

    if context is not None:
        context.advice_category = advice_category

    if context is None:
        context = ExecutionContext(advice_category=advice_category)

    advisor = get_advisor(
        context,
        target_profile,
        model,
        optimization_targets=optimization_targets,
        optimization_profile=optimization_profile,
        backends=backends,
    )
    advisor.run(context)


def get_advisor(
    context: ExecutionContext,
    target_profile: str | Path,
    model: str | Path,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Find appropriate advisor for the target."""
    if extra_args.get("optimization_profile"):
        extra_args["optimization_profile"] = get_optimization_profile(
            extra_args["optimization_profile"]
        )
    target = profile(target_profile).target
    factory_function = target_registry.items[target].advisor_factory_func
    return factory_function(
        context,
        target_profile,
        model,
        **extra_args,
    )
