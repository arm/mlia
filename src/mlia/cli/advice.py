# Copyright 2021, Arm Ltd.
"""Module for the advice generation."""
import logging
import math
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
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
    OPTIMIZATION = 3


class OptimizationResults(TypedDict):
    """Class represents results of the optimization."""

    perf_metrics: pd.DataFrame
    optimization_type: str
    optimization_target: Union[int, float]


class AdvisorContext(TypedDict, total=False):
    """Inference advisor context."""

    operators: Operators
    perf_metrics: PerformanceMetrics
    optimization_results: OptimizationResults
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


def advice_optimization(ctx: AdvisorContext) -> List[str]:
    """Advice to try model optimization."""
    return [
        "Check if you can improve the performance by applying "
        "tooling techniques to your model.",
        "Note: you will need a Keras/TF.saved_model input for that.",
        "For example: mlia optimization --optimization-type "
        "pruning --optimization-target 0.5 /path/to/keras_model",
        "For more info: mlia optimization --help",
    ]


def advice_npu_support(ctx: AdvisorContext) -> List[str]:
    """General NPU operators compatibility advice."""
    return [
        "For better performance, make sure that all the operators of your "
        "final TFLite model are supported by the NPU.",
        "For more details, run: mlia operators --help",
    ]


def next_optimization_target(
    opt_target: Union[int, float], opt_type: str
) -> Union[float, int]:
    """Calculate next optimization target value."""
    if opt_type == "pruning":
        return min(opt_target + 0.05, 0.95)

    # return next nearest power of two for clustering
    return min(int(2 ** int(math.log(opt_target, 2) + 1)), 128)


def get_metrics(
    data: pd.DataFrame, improved: bool, metrics: List[str]
) -> List[Tuple[str, int]]:
    """Filter and sort metrics."""
    impr_metric_name = "Improvement (%)"

    metric_values = (
        data[data[impr_metric_name] > 0]
        if improved
        else data[data[impr_metric_name] < 0]
    )

    return sorted(
        (
            (metric, value)
            for metric, value in metric_values[impr_metric_name].iteritems()
            if metric in metrics
        ),
        key=lambda row: metrics.index(row[0]),
    )


def advice_optimization_improvement(ctx: AdvisorContext) -> List[str]:
    """Advice on results of optimization."""
    optimization_results = ctx.get("optimization_results")
    if not optimization_results:
        return []

    perf_metrics: pd.DataFrame = optimization_results.get("perf_metrics")

    opt_type = optimization_results["optimization_type"]
    opt_target = optimization_results["optimization_target"]
    if isinstance(opt_target, float) and opt_target.is_integer():
        opt_target = int(opt_target)

    result = [
        f"With the selected optimization (type: {opt_type} - target: {opt_target})"
    ]

    metrics = [
        "SRAM used (KiB)",
        "DRAM used (KiB)",
        "On chip flash used (KiB)",
        "Off chip flash used (KiB)",
        "NPU total cycles",
    ]

    improved_metrics = get_metrics(perf_metrics, True, metrics)
    degraded_metrics = get_metrics(perf_metrics, False, metrics)

    impr_text = [
        f"- You have achieved {value:.2f}% performance improvement in {metric}"
        for metric, value in improved_metrics
    ]

    degr_text = [
        f"- {metric} have degraded by {abs(value):.2f}%"
        for metric, value in degraded_metrics
    ]

    result = result + impr_text + degr_text
    if impr_text:
        next_opt_target = next_optimization_target(opt_target, opt_type)
        if opt_target <= next_opt_target:
            result.append(
                "You can try to push higher the optimization target "
                f"(e.g. {next_opt_target}) "
                "to check if that can be further improved."
            )
    elif degr_text:
        result.append(
            "The performance seems to have degraded after "
            "applying the selected optimizations, "
            "try exploring different optimization types/targets."
        )

    return result


def show_advice(
    ctx: AdvisorContext,
    advice_group: Optional[Union[AdviceGroup, List[AdviceGroup]]] = None,
) -> None:
    """Show advice based on provided data."""
    advice_producers = {
        AdviceGroup.OPERATORS_COMPATIBILITY: [
            advice_non_npu_operators,
            advice_unsupported_operators,
            advice_all_operators_supported,
        ],
        AdviceGroup.PERFORMANCE: [
            advice_increase_operator_compatibility,
            advice_optimization,
        ],
        AdviceGroup.OPTIMIZATION: [advice_optimization_improvement, advice_npu_support],
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
    valuable_advices = ("\n".join(item) for item in responses if len(item) > 0)

    for i, advice in enumerate(valuable_advices, start=1):
        if i != 1:
            LOGGER.info("")
        LOGGER.info(tabulate([(i, advice)], tablefmt="plain"))

    LOGGER.info("")
