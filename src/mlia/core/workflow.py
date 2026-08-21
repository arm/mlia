# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Workflow executors."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, Sequence

from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.core.data_analysis import DataAnalyzer, PatternAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.mixins import ContextMixin
from mlia.core.output_collection import StandardizedOutputCollector


class WorkflowExecutor(ABC):
    """Base workflow executor."""

    @abstractmethod
    def run(self) -> dict[str, Any] | None:
        """Run the workflow and return canonical standardized output, if produced."""


class DefaultWorkflowExecutor(WorkflowExecutor):
    """Default workflow executor."""

    def __init__(
        self,
        context: Context,
        collectors: Sequence[DataCollector],
        analyzers: Sequence[DataAnalyzer],
        pattern_analyzers: Sequence[PatternAnalyzer] | None = None,
        max_pattern_passes: int = 5,
    ):
        """Init default workflow executor."""
        self.context = context
        self.collectors = collectors
        self.analyzers = analyzers
        self.pattern_analyzers = pattern_analyzers or []
        self.max_pattern_passes = max_pattern_passes
        self.output_collector = StandardizedOutputCollector()

    def run(self) -> dict[str, Any] | None:
        """Run the workflow."""
        self.inject_context()

        collected_data = self.collect_data()
        analyzed_data = self.analyze_data(collected_data)
        if self.pattern_analyzers:
            self.detect_patterns(analyzed_data)

        return self.output_collector.build_output()

    def collect_data(self) -> list[DataItem]:
        """Run data collectors."""
        collected_data = []
        for collector in self.collectors:
            try:
                if (data_item := collector.collect_data()) is not None:
                    collected_data.append(data_item)
                    self.output_collector.submit_data_item(data_item)
            except FunctionalityNotSupportedError:
                continue

        return collected_data

    def analyze_data(self, collected_data: list[DataItem]) -> list[DataItem]:
        """Run data analyzers."""
        analyzed_data = []
        for analyzer in self.analyzers:
            for item in collected_data:
                analyzer.analyze_data(item)

            for data_item in analyzer.get_analyzed_data():
                analyzed_data.append(data_item)
                self.output_collector.submit_data_item(data_item)
        return analyzed_data

    def detect_patterns(self, analyzed_data: list[DataItem]) -> list[DataItem]:
        """Detect patterns in analyzed facts."""
        all_facts = list(analyzed_data)
        pass_number = 0

        while pass_number < self.max_pattern_passes:
            pass_number += 1
            new_facts_this_pass = 0

            for analyzer in self.pattern_analyzers:
                if hasattr(analyzer, "clear_cache"):
                    analyzer.clear_cache()

                new_patterns = analyzer.analyze_patterns(all_facts)

                for pattern_fact in new_patterns:
                    all_facts.append(pattern_fact)
                    new_facts_this_pass += 1

                if hasattr(analyzer, "detected_patterns"):
                    analyzer.detected_patterns = new_patterns

            if new_facts_this_pass == 0:
                break

        return all_facts

    def inject_context(self) -> None:
        """Inject context object into context-aware components."""
        context_aware_components = (
            comp
            for comp in itertools.chain(
                self.collectors,
                self.analyzers,
                self.pattern_analyzers,
            )
            if isinstance(comp, ContextMixin)
        )

        for component in context_aware_components:
            component.set_context(self.context)
