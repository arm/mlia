# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the API functions."""
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from mlia.core._typing import PathOrFileLike
from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.devices.ethosu.advisor import configure_and_get_ethosu_advisor
from mlia.devices.tosa.advisor import configure_and_get_tosa_advisor
from mlia.utils.filesystem import get_target


logger = logging.getLogger(__name__)


def get_advice(
    target_profile: str,
    model: Union[Path, str],
    category: Literal["all", "operators", "performance", "optimization"] = "all",
    optimization_targets: Optional[List[Dict[str, Any]]] = None,
    working_dir: Union[str, Path] = "mlia_output",
    output: Optional[PathOrFileLike] = None,
    context: Optional[ExecutionContext] = None,
    backends: Optional[List[str]] = None,
) -> None:
    """Get the advice.

    This function represents an entry point to the library API.

    Based on provided parameters it will collect and analyze the data
    and produce the advice.

    :param target_profile: target profile identifier
    :param model: path to the NN model
    :param category: category of the advice. MLIA supports four categories:
           "all", "operators", "performance", "optimization". If not provided
           category "all" is used by default.
    :param optimization_targets: optional model optimization targets that
           could be used for generating advice in categories
           "all" and "optimization."
    :param working_dir: path to the directory that will be used for storing
           intermediate files during execution (e.g. converted models)
    :param output: path to the report file. If provided MLIA will save
           report in this location. Format of the report automatically
           detected based on file extension.
    :param context: optional parameter which represents execution context,
           could be used for advanced use cases
    :param backends: A list of backends that should be used for the given
           target. Default settings will be used if None.

    Examples:
        NB: Before launching MLIA, the logging functionality should be configured!

        Getting the advice for the provided target profile and the model

        >>> get_advice("ethos-u55-256", "path/to/the/model")

        Getting the advice for the category "performance" and save result report in file
        "report.json"

        >>> get_advice("ethos-u55-256", "path/to/the/model", "performance",
                       output="report.json")

    """
    advice_category = AdviceCategory.from_string(category)

    if context is not None:
        context.advice_category = advice_category

    if context is None:
        context = ExecutionContext(
            advice_category=advice_category,
            working_dir=working_dir,
        )

    advisor = get_advisor(
        context,
        target_profile,
        model,
        output,
        optimization_targets=optimization_targets,
        backends=backends,
    )

    advisor.run(context)


def get_advisor(
    context: ExecutionContext,
    target_profile: str,
    model: Union[Path, str],
    output: Optional[PathOrFileLike] = None,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Find appropriate advisor for the target."""
    target_factories = {
        "ethos-u55": configure_and_get_ethosu_advisor,
        "ethos-u65": configure_and_get_ethosu_advisor,
        "tosa": configure_and_get_tosa_advisor,
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
        output,
        **extra_args,
    )
