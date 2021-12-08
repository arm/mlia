# Copyright 2021, Arm Ltd.
"""Module for the advice generation."""
import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import Union

import pandas as pd
from mlia.cli.options import get_device_opts
from mlia.devices.ethosu.metadata import Operators
from mlia.devices.ethosu.metrics import PerformanceMetrics
from mlia.nn.tensorflow.utils import is_keras_model


@dataclass
class Advice:
    """IA advice."""

    advice_msgs: List[str]


class AdviceGroup(Enum):
    """Advice group."""

    OPERATORS_COMPATIBILITY = 1
    PERFORMANCE = 2
    OPTIMIZATION = 3
    COMMON = 4


class OptimizationResults(TypedDict):
    """Class represents results of the optimization."""

    perf_metrics: pd.DataFrame
    optimizations: List[Tuple[str, Union[int, float]]]


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


def advice_all_operators_supported(
    ctx: AdvisorContext, recommend_estimate_performance: bool = True
) -> List[str]:
    """Advice if all operators supported."""
    operators = ctx.get("operators")
    if not operators or operators.npu_unsupported_ratio != 0:
        return []

    device_opts = " ".join(get_device_opts(ctx.get("device_args")))
    if device_opts:
        device_opts = " " + device_opts
    model_opts = ctx.get("model")

    result = [
        "You don't have any unsupported operators, your model will "
        "run completely on NPU."
    ]
    if recommend_estimate_performance:
        result += [
            "Check the estimated performance by running the following command:",
            f"mlia performance{device_opts} {model_opts}",
        ]
    return result


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
    model_path = "/path/to/keras_model"
    keras_note = ["Note: you will need a Keras/TF.saved_model input for that."]

    if (model := ctx.get("model")) and is_keras_model(Path(model)):
        model_path = model
        keras_note = []

    return (
        [
            "Check if you can improve the performance by applying "
            "tooling techniques to your model."
        ]
        + keras_note
        + [
            "For example: mlia optimization --optimization-type "
            f"pruning,clustering --optimization-target 0.5,32 {model_path}",
            "For more info: mlia optimization --help",
        ]
    )


def advice_npu_support(_ctx: AdvisorContext) -> List[str]:
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
        return round(min(opt_target + 0.1, 0.9), 2)

    if opt_type == "clustering":
        # return next lowest power of two for clustering
        next_target = math.log(opt_target, 2)
        if next_target.is_integer():
            next_target -= 1

        return max(int(2 ** int(next_target)), 4)

    raise Exception(f"Unknown optimization type {opt_type}")


def get_metrics(
    optimization_results: OptimizationResults, improved: bool
) -> List[Tuple[str, int]]:
    """Filter and sort metrics."""
    perf_metrics: pd.DataFrame = optimization_results.get("perf_metrics")

    metrics = [
        "SRAM used (KiB)",
        "DRAM used (KiB)",
        "On chip flash used (KiB)",
        "Off chip flash used (KiB)",
        "NPU total cycles",
    ]

    impr_metric_name = "Improvement (%)"

    metric_values = (
        perf_metrics[perf_metrics[impr_metric_name] > 0]
        if improved
        else perf_metrics[perf_metrics[impr_metric_name] < 0]
    )

    return sorted(
        (
            (metric, value)
            for metric, value in metric_values[impr_metric_name].iteritems()
            if metric in metrics
        ),
        key=lambda row: metrics.index(row[0]),
    )


def advice_optimization_improvement(
    ctx: AdvisorContext, recommend_run_optimizations: bool = False
) -> List[str]:
    """Advice on results of optimization."""
    optimization_results = ctx.get("optimization_results")
    if not optimization_results:
        return []

    optimizations = [
        (
            opt_type,
            int(opt_target)
            if isinstance(opt_target, float) and opt_target.is_integer()
            else opt_target,
        )
        for opt_type, opt_target in optimization_results.get("optimizations", [])
    ]
    if not optimizations:
        return []

    opt_text = " + ".join(
        f"{opt_type}: {opt_target}" for opt_type, opt_target in optimizations
    )

    result = [f"With the selected optimization ({opt_text})"]

    impr_text = [
        f"- You have achieved {value:.2f}% performance improvement in {metric}"
        for metric, value in get_metrics(optimization_results, improved=True)
    ]

    degr_text = [
        f"- {metric} have degraded by {abs(value):.2f}%"
        for metric, value in get_metrics(optimization_results, improved=False)
    ]

    result = result + impr_text + degr_text

    if impr_text:
        next_opt_targets = {
            opt_type: new_target
            for opt_type, opt_target, new_target in (
                (opt_type, opt_target, next_optimization_target(opt_target, opt_type))
                for opt_type, opt_target in optimizations
            )
            if (
                (opt_type == "pruning" and opt_target < new_target)
                or (opt_type == "clustering" and opt_target > new_target)
            )
        }

        if next_opt_targets:
            next_opt_targets_text = " and/or ".join(
                f"{opt_type} {opt_target}"
                for opt_type, opt_target in next_opt_targets.items()
            )
            result.append(
                "You can try to push the optimization target higher "
                f"(e.g. {next_opt_targets_text}) "
                "to check if those results can be further improved."
            )
            if recommend_run_optimizations:
                device_opts = " ".join(get_device_opts(ctx.get("device_args")))
                if device_opts:
                    device_opts = f" {device_opts}"
                model_opts = ctx.get("model")

                result.append("For more info, see: mlia optimization --help")

                new_opt = ",".join(next_opt_targets.keys())
                new_target = ",".join(str(v) for v in next_opt_targets.values())
                result.append(
                    f"Optimization command: "
                    f"mlia optimization --optimization-type {new_opt} "
                    f"--optimization-target {new_target}{device_opts} {model_opts}"
                )
    elif degr_text:
        result.append(
            "The performance seems to have degraded after "
            "applying the selected optimizations, "
            "try exploring different optimization types/targets."
        )

    return result


def advice_hyperparameter_tuning(_ctx: AdvisorContext) -> List[str]:
    """Advice on hyperparameter tuning."""
    return [
        "The applied tooling techniques have an impact "
        "on accuracy. Additional hyperparameter tuning may be required "
        "after any optimization."
    ]


advice_all_operators_supported_no_commands = partial(
    advice_all_operators_supported, recommend_estimate_performance=False
)

advice_optimization_improvement_extended = partial(
    advice_optimization_improvement, recommend_run_optimizations=True
)


def get_advice_producers() -> Dict[AdviceGroup, List[Callable]]:
    """Return advice producers grouped by category."""
    return {
        AdviceGroup.OPERATORS_COMPATIBILITY: [
            advice_non_npu_operators,
            advice_unsupported_operators,
            advice_all_operators_supported,
        ],
        AdviceGroup.PERFORMANCE: [
            advice_increase_operator_compatibility,
            advice_optimization,
        ],
        AdviceGroup.OPTIMIZATION: [
            advice_optimization_improvement_extended,
            advice_npu_support,
        ],
        AdviceGroup.COMMON: [
            advice_non_npu_operators,
            advice_unsupported_operators,
            advice_all_operators_supported_no_commands,
            advice_optimization_improvement_extended,
            advice_hyperparameter_tuning,
        ],
    }


def filter_advice_producers(
    advice_group: Optional[Union[AdviceGroup, List[AdviceGroup]]] = None
) -> List[Callable]:
    """Filter advice producers based on provided parameters."""
    if isinstance(advice_group, AdviceGroup):
        advice_group = [advice_group]

    return [
        advice_producer
        for adv_group, adv_group_producers in get_advice_producers().items()
        for advice_producer in adv_group_producers
        if advice_group is None or adv_group in advice_group
    ]


def produce_advice(
    ctx: AdvisorContext,
    advice_group: Optional[Union[AdviceGroup, List[AdviceGroup]]],
) -> List[Advice]:
    """Produce advice based on provided data."""
    selected_advice_producers = filter_advice_producers(advice_group)

    return [
        Advice(advice)
        for advice in (
            advice_producer(ctx) for advice_producer in selected_advice_producers
        )
        if advice
    ]
