# Copyright 2022, Arm Ltd.
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
from mlia.core.events import EventHandler
from mlia.devices.ethosu.advisor import EthosUInferenceAdvisor
from mlia.devices.ethosu.handlers import EthosUEventHandler


logger = logging.getLogger(__name__)


_DEFAULT_OPTIMIZATION_TARGETS = [
    {
        "optimization_type": "pruning",
        "optimization_target": 0.5,
        "layers_to_optimize": None,
    },
    {
        "optimization_type": "clustering",
        "optimization_target": 32,
        "layers_to_optimize": None,
    },
]


def get_advice(
    target: str,
    model: Union[Path, str],
    category: Literal["all", "operators", "performance", "optimization"] = "all",
    optimization_targets: Optional[List[Dict[str, Any]]] = None,
    working_dir: Union[str, Path] = "mlia_output",
    output: Optional[PathOrFileLike] = None,
    context: Optional[ExecutionContext] = None,
) -> None:
    """Get the advice.

    This function represents an entry point to the library API.

    Based on provided parameters it will collect and analyze the data
    and produce the advice.

    :param target: target profile identifier
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


    Examples:
        NB: Before launching MLIA, the logging functionality should be configured!

        Getting the advice for the provided target profile and the model

        >>> get_advice("U55-256", "path/to/the/model")

        Getting the advice for the category "performance" and save result report in file
        "report.json"

        >>> get_advice("U55-256", "path/to/the/model", "performance",
                       output="report.json")

    """
    advice_category = AdviceCategory.from_string(category)
    config_parameters = _get_config_parameters(target, model, optimization_targets)
    event_handlers = _get_event_handlers(output)

    if context is not None:
        if context.advice_category is None:
            context.advice_category = advice_category

        if context.config_parameters is None:
            context.config_parameters = config_parameters

        if context.event_handlers is None:
            context.event_handlers = event_handlers

    if context is None:
        context = ExecutionContext(
            advice_category=advice_category,
            working_dir=working_dir,
            config_parameters=config_parameters,
            event_handlers=event_handlers,
        )

    advisor = _get_advisor(target)
    advisor.run(context)


def _get_advisor(target: Optional[str]) -> InferenceAdvisor:
    """Find appropriate advisor for the target."""
    if not target:
        raise Exception("Target is not provided")

    return EthosUInferenceAdvisor()


def _get_config_parameters(
    target: str,
    model: Union[Path, str],
    optimization_targets: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Get configuration parameters for the advisor."""
    advisor_parameters: Dict[str, Any] = {
        "ethos_u_inference_advisor": {
            "model": model,
            "device": {
                "target": target,
            },
        },
    }

    if not optimization_targets:
        optimization_targets = _DEFAULT_OPTIMIZATION_TARGETS

    advisor_parameters.update(
        {
            "ethos_u_model_optimizations": {
                "optimizations": [
                    optimization_targets,
                ],
            },
        }
    )

    return advisor_parameters


def _get_event_handlers(output: Optional[PathOrFileLike]) -> List[EventHandler]:
    """Return list of the event handlers."""
    return [EthosUEventHandler(output)]
