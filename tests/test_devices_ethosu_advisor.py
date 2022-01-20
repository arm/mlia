# Copyright 2022, Arm Ltd.
"""Tests for Ethos-U IA module."""
from mlia.devices.ethosu.advisor import EthosUInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert EthosUInferenceAdvisor.name() == "ethos_u_inference_advisor"
