# Copyright 2021, Arm Ltd.
"""Module for the advice generation."""
import logging
from enum import Enum
from textwrap import wrap
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from mlia.cli.options import get_device_opts
from mlia.metadata import Operators
from mlia.metrics import PerformanceMetrics
from tabulate import tabulate
from typing_extensions import TypedDict


LOGGER = logging.getLogger("mlia.cli")


class AdviceGroup(Enum):
    """Advice group."""

    OPERATORS_COMPATIBILITY = 1
    PERFORMANCE = 2


class AdvisorContext(TypedDict, total=False):
    """Inference advisor context."""

    operators: Operators
    perf_metrics: PerformanceMetrics
    device_args: Dict
    model: str


def advice_non_npu_operators(ctx: AdvisorContext) -> List[str]:
    """Advice for the non NPU operators."""
    operators = ctx.get("operators")
    if not operators or operators.npu_unsupported_ratio == 0:
        return []

    return [
        f"You have {operators.npu_unsupported_ratio*100:.0f}% of operators "
        "that cannot be placed on the NPU.",
        "For better performance, please review the reasons reported in the table, "
        "and adjust the model accordingly where possible.",
    ]


def advice_unsupported_operators(ctx: AdvisorContext) -> List[str]:
    """Advice for the unsupported operators."""
    operators = ctx.get("operators")

    if operators:
        cpu_only = [op.op_type for op in operators.ops if op.cpu_only]

        if cpu_only:
            cpu_only_ops = ",".join(sorted(set(cpu_only)))
            cpu_only_ops_num = len(cpu_only)
            return [
                f"You have at least {len(cpu_only)} "
                f"operator{'s' if cpu_only_ops_num > 1 else ''} that is CPU "
                f"only: {cpu_only_ops}.",
                "Using operators that are supported by the NPU will "
                "improve performance.",
                "For guidance on supported operators, run: mlia operators "
                "--supported-ops-report",
            ]

    return []


def advice_all_operators_supported(ctx: AdvisorContext) -> List[str]:
    """Advice if all operators supported."""
    operators = ctx.get("operators")
    if not operators or operators.npu_unsupported_ratio != 0:
        return []

    device_opts = " ".join(get_device_opts(ctx.get("device_args")))
    if device_opts:
        device_opts = " " + device_opts
    model_opts = ctx.get("model")

    return [
        "You don't have any unsupported operators, your model will "
        "run completely on NPU.",
        "Check the estimated performance by running the following command:",
        f"mlia performance{device_opts} {model_opts}",
    ]


def advice_increase_operator_compatibility(ctx: AdvisorContext) -> List[str]:
    """Advice to increase op compatibility for performance improvement."""
    device_opts = " ".join(get_device_opts(ctx.get("device_args")))
    if device_opts:
        device_opts = " " + device_opts
    model_opts = ctx.get("model")

    return [
        "You can improve the inference time by using only operators "
        "that are supported by the NPU.",
        "Try running the following command to verify that:",
        f"mlia operators{device_opts} {model_opts}",
    ]


def advice_model_optimization(ctx: AdvisorContext) -> List[str]:
    """Advice to try model optimization."""
    return [
        "Check if you can improve the performance by applying "
        "tooling techniques to your model.",
        "Note: you will need a Keras/TF.saved_model input for that.",
        "For example:  mlia model_optimization --optimization-type "
        "pruning --optimization-target 0.5 /path/to/keras_model",
        "For more info: mlia model_optimization --help",
    ]


def show_advice(
    ctx: AdvisorContext,
    advice_group: Optional[Union[AdviceGroup, List[AdviceGroup]]] = None,
) -> None:
    """Show advice based on provided data."""
    LOGGER.info(
        """
=== Advice Generation =========================================================
"""
    )

    advice_producers = {
        AdviceGroup.OPERATORS_COMPATIBILITY: [
            advice_non_npu_operators,
            advice_unsupported_operators,
            advice_all_operators_supported,
        ],
        AdviceGroup.PERFORMANCE: [
            advice_increase_operator_compatibility,
            advice_model_optimization,
        ],
    }

    if isinstance(advice_group, AdviceGroup):
        advice_group = [advice_group]

    selected_advice_producers = [
        advice_producer
        for adv_group, adv_group_producers in advice_producers.items()
        for advice_producer in adv_group_producers
        if advice_group is None or adv_group in advice_group
    ]

    responses = (advice_producer(ctx) for advice_producer in selected_advice_producers)
    valuable_advices = (
        "\n".join(
            wrapped_line
            for line in item
            # do not wrap if line contains mlia command
            for wrapped_line in wrap(line, 100 if "mlia" not in line else 1000)
        )
        for item in responses
        if len(item) > 0
    )

    for i, advice in enumerate(valuable_advices, start=1):
        if i != 1:
            LOGGER.info("")
        LOGGER.info(tabulate([(i, advice)], tablefmt="plain"))

    LOGGER.info("")
