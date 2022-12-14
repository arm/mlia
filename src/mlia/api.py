# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the API functions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.cortex_a.advisor import configure_and_get_cortexa_advisor
from mlia.target.ethos_u.advisor import configure_and_get_ethosu_advisor
from mlia.target.tosa.advisor import configure_and_get_tosa_advisor
from mlia.utils.filesystem import get_target

logger = logging.getLogger(__name__)


def get_advice(
    target_profile: str,
    model: str | Path,
    category: set[str],
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
    target_factories = {
        "ethos-u55": configure_and_get_ethosu_advisor,
        "ethos-u65": configure_and_get_ethosu_advisor,
        "tosa": configure_and_get_tosa_advisor,
        "cortex-a": configure_and_get_cortexa_advisor,
    }

    try:
        target = get_target(target_profile)
        factory_function = target_factories[target]
    except KeyError as err:
        raise Exception(f"Unsupported profile {target_profile}") from err

    return factory_function(
        context,
        target_profile,
        model,
        **extra_args,
    )
