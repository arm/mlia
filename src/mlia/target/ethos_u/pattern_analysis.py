# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U pattern analysis module for detecting optimization opportunities."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import PatternAnalyzer
from mlia.core.data_analysis import register_fact_type
from mlia.target.ethos_u.data_analysis import EthosULayerUsesLUT


@register_fact_type(
    "ethos_u_ineffective_activation_pattern",
    "pattern",
    "Multiple layers using LUT-based activations that could be optimized",
)
@dataclass
class IneffectiveActivationPattern(Fact):
    """Pattern indicating multiple layers with inefficient activations.

    This composite fact is generated when multiple layers use LUT-based
    activations (like SOFTMAX) that run on CPU instead of NPU.
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

    Detects when multiple layers use LUT-based activations that could
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

        # Filter for LUT-related facts
        lut_facts = [f for f in facts if isinstance(f, EthosULayerUsesLUT)]

        # Group by activation type
        activation_groups: dict[str, list] = {}
        for fact in lut_facts:
            if hasattr(fact, "activation_type"):
                act_type = fact.activation_type
                if act_type not in activation_groups:
                    activation_groups[act_type] = []
                activation_groups[act_type].append(fact)

        detected_patterns: list[Fact] = []

        # Check for patterns in each activation type
        for act_type, layer_facts in activation_groups.items():
            if len(layer_facts) >= 1:  # Even single LUT usage is worth noting
                # Collect affected layer information
                affected_layers = []
                for fact in layer_facts:
                    if hasattr(fact, "operator_name"):
                        affected_layers.append(
                            f"{fact.operator_name} ({fact.location})"
                        )

                # Generate recommendation based on activation type
                if act_type.upper() in ["SOFTMAX", "SIGMOID", "TANH"]:
                    recommendation = (
                        f"Consider replacing {act_type} "
                        "with NPU-friendly alternatives."
                    )
                else:
                    recommendation = (
                        f"Layer uses {act_type} operation which is not "
                        "easily quantized. Consider alternative activation "
                        "functions that can be easily quantized."
                    )

                pattern = IneffectiveActivationPattern(
                    affected_layers=affected_layers,
                    layer_count=len(layer_facts),
                    activation_types=[act_type],
                    recommendation=recommendation,
                )
                detected_patterns.append(pattern)

        return detected_patterns
