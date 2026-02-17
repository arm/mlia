# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U pattern analysis module for detecting optimization opportunities."""

from __future__ import annotations

from dataclasses import dataclass, field

from mlia.core.data_analysis import Fact, PatternAnalyzer, register_fact_type
from mlia.target.ethos_u.data_analysis import EthosULayerSuboptimalActivation


@register_fact_type(
    "ethos_u_ineffective_activation_pattern",
    "pattern",
    "Multiple layers using ineffective activations that could be optimized",
)
@dataclass
class IneffectiveActivationPattern(Fact):
    """Pattern indicating multiple layers with inefficient activations.

    This composite fact is generated when multiple layers use
    suboptimal activation functions that are hard to quantize effectively.
    """

    affected_layers: list[str] = field(default_factory=list)
    layer_count: int = 0
    activation_types: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = super().to_dict()
        result.update(
            {
                "affected_layers": self.affected_layers,
                "layer_count": self.layer_count,
                "activation_types": self.activation_types,
                "recommendation": self.recommendation,
            }
        )
        return result


class ActivationFunctionPatternAnalyzer(PatternAnalyzer):
    """Analyzes activation function patterns across layers.

    Detects when multiple layers use suboptimal activation functions that could
    be replaced with more efficient alternatives or quantized differently.
    """

    def has_already_generated_patterns(self, facts: list[Fact]) -> bool:
        """Check if we've already generated ineffective activation patterns.

        :param facts: List of facts to check
        :return: True if patterns already exist, False otherwise
        """
        # Check if any IneffectiveActivationPattern facts already exist
        return any(isinstance(f, IneffectiveActivationPattern) for f in facts)

    def analyze_patterns(self, facts: list[Fact]) -> list[Fact]:
        """Analyze facts to detect inefficient activation patterns.

        :param facts: List of all facts from analysis
        :return: List of newly detected pattern facts
        """
        # Skip analysis if we've already generated patterns
        if self.has_already_generated_patterns(facts):
            return []

        bad_activation_facts = [
            f for f in facts if isinstance(f, EthosULayerSuboptimalActivation)
        ]

        if not bad_activation_facts:
            return []

        affected_layers = []
        activation_types_set = set()

        for fact in bad_activation_facts:
            if hasattr(fact, "operator_name"):
                affected_layers.append(f"{fact.operator_name} ({fact.location})")
            if hasattr(fact, "activation_type"):
                activation_types_set.add(fact.activation_type)

        # Create single pattern for all suboptimal activations
        activation_types = sorted(activation_types_set)

        recommendation = (
            "Consider replacing suboptimal activation functions with "
            "NPU-friendly alternatives."
        )

        # Identify activation functions that use unsupported operations
        unsupported_activations = {"MISH", "SELU", "SOFTPLUS"}
        used_unsupported_activations = unsupported_activations & set(activation_types)

        if used_unsupported_activations:
            recommendation += (
                "\nThe following activation functions use unsupported operations: "
                + ", ".join(sorted(used_unsupported_activations))
            )

        pattern = IneffectiveActivationPattern(
            affected_layers=affected_layers,
            layer_count=len(bad_activation_facts),
            activation_types=activation_types,
            recommendation=recommendation,
        )

        return [pattern]
