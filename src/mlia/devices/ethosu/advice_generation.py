# Copyright 2021, Arm Ltd.
"""Ethos-U advice generation."""
from functools import singledispatchmethod
from typing import List
from typing import Union

from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import ContextAwareAdviceProducer
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.devices.ethosu.data_analysis import AllOperatorsSupportedOnNPU
from mlia.devices.ethosu.data_analysis import HasCPUOnlyOperators
from mlia.devices.ethosu.data_analysis import HasUnsupportedOnNPUOperators
from mlia.devices.ethosu.data_analysis import OptimizationResults
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings


class EthosUAdviceProducer(FactBasedAdviceProducer):
    """Ethos-U advice producer."""

    @singledispatchmethod
    def produce_advice(self, data_item: DataItem) -> None:
        """Produce advice."""

    @produce_advice.register
    @advice_category(AdviceCategory.OPERATORS, AdviceCategory.ALL)
    def handle_cpu_only_ops(self, data_item: HasCPUOnlyOperators) -> None:
        """Advice for CPU only operators."""
        cpu_only_ops = ",".join(sorted(set(data_item.cpu_only_ops)))
        cpu_only_ops_num = len(data_item.cpu_only_ops)

        self.add_advice(
            [
                f"You have at least {cpu_only_ops_num} "
                f"operator{'s' if cpu_only_ops_num > 1 else ''} that is CPU "
                f"only: {cpu_only_ops}.",
                "Using operators that are supported by the NPU will "
                "improve performance.",
            ]
            + self.context.action_resolver.supported_operators_info()
        )

    @produce_advice.register
    @advice_category(AdviceCategory.OPERATORS, AdviceCategory.ALL)
    def handle_unsupported_operators(
        self, data_item: HasUnsupportedOnNPUOperators
    ) -> None:
        """Advice for the unsupported operators."""
        self.add_advice(
            [
                f"You have {data_item.npu_unsupported_ratio*100:.0f}% of operators "
                "that cannot be placed on the NPU.",
                "For better performance, please review the reasons reported "
                "in the table, and adjust the model accordingly "
                "where possible.",
            ]
        )

    @produce_advice.register
    @advice_category(AdviceCategory.OPERATORS, AdviceCategory.ALL)
    def handle_all_operators_supported(
        self, data_item: AllOperatorsSupportedOnNPU  # pylint: disable=unused-argument
    ) -> None:
        """Advice if all operators supported."""
        self.add_advice(
            [
                "You don't have any unsupported operators, your model will "
                "run completely on NPU."
            ]
            + self.context.action_resolver.check_performance()
        )

    @produce_advice.register
    @advice_category(AdviceCategory.OPTIMIZATION, AdviceCategory.ALL)
    def handle_optimization_results(self, data_item: OptimizationResults) -> None:
        """Advice based on optimization results."""
        if not data_item.diffs or len(data_item.diffs) != 1:
            return

        optim_details = data_item.diffs[0]
        metrics = [
            (metric_name, metric_value)
            for (metric_name, metric_value) in [
                ("DRAM used (KB)", optim_details.dram),
                ("SRAM used (KB)", optim_details.sram),
                ("On chip flash used (KB)", optim_details.on_chip_flash),
                ("Off chip flash used (KB)", optim_details.off_chip_flash),
                ("NPU total cycles", optim_details.npu_total_cycles),
            ]
            if not metric_value.same
        ]

        improved = [
            f"- You have achieved {abs(metric_value.diff):.2f}% performance "
            f"improvement in {metric_name}"
            for metric_name, metric_value in metrics
            if metric_value.improved
        ]

        degraded = [
            f"- {metric_name} have degraded by {abs(metric_value.diff):.2f}%"
            for metric_name, metric_value in metrics
            if metric_value.degraded
        ]

        opts = ", ".join(str(s) for s in optim_details.opt_type)
        messages = [f"With the selected optimization ({opts})", *improved, *degraded]

        if improved:
            if next_optimization_target := self.get_next_optimization_targets(
                optim_details.opt_type
            ):
                next_optimization_target_as_str = " and/or ".join(
                    str(item) for item in next_optimization_target
                )

                messages.append(
                    "You can try to push the optimization target higher "
                    f"(e.g. {next_optimization_target_as_str}) "
                    "to check if those results can be further improved."
                )
                messages += self.context.action_resolver.apply_optimizations(
                    opt_settings=next_optimization_target
                )

        elif degraded:
            messages.append(
                "The performance seems to have degraded after "
                "applying the selected optimizations, "
                "try exploring different optimization types/targets."
            )

        self.add_advice(messages)

        self.add_advice(
            [
                "The applied tooling techniques have an impact "
                "on accuracy. Additional hyperparameter tuning may be required "
                "after any optimization."
            ]
        )

    @staticmethod
    def get_next_optimization_targets(
        opt_type: List[OptimizationSettings],
    ) -> List[OptimizationSettings]:
        """Get next optimization targets."""
        next_targets = (item.next_target() for item in opt_type)

        # filter out targets that have not been changed
        valid_targets = [
            next_
            for next_, old in zip(next_targets, opt_type)
            if (
                old.optimization_type == "pruning"
                and old.optimization_target < next_.optimization_target
            )
            or (
                old.optimization_type == "clustering"
                and old.optimization_target > next_.optimization_target
            )
        ]
        return valid_targets


class EthosUStaticAdviceProducer(ContextAwareAdviceProducer):
    """Advice producer that not depends on input data."""

    def produce_advice(self, data_item: DataItem) -> None:
        """Do not process passed data items."""

    def get_advice(self) -> Union[Advice, List[Advice]]:
        """Return predefined advice based on category."""
        if self.context.advice_category is None:
            return []

        advice_per_category = {
            AdviceCategory.PERFORMANCE: [
                Advice(
                    [
                        "You can improve the inference time by using only operators "
                        "that are supported by the NPU.",
                    ]
                    + self.context.action_resolver.check_operator_compatibility()
                ),
                Advice(
                    [
                        "Check if you can improve the performance by applying "
                        "tooling techniques to your model."
                    ]
                    + self.context.action_resolver.apply_optimizations()
                ),
            ],
            AdviceCategory.OPTIMIZATION: [
                Advice(
                    [
                        "For better performance, make sure that all the operators "
                        "of your final TFLite model are supported by the NPU.",
                    ]
                    + self.context.action_resolver.operator_compatibility_details()
                )
            ],
        }

        return advice_per_category.get(self.context.advice_category, [])
