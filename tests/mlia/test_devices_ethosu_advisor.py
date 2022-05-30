# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U MLIA module."""
from mlia.devices.ethosu.advisor import EthosUInferenceAdvisor


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert EthosUInferenceAdvisor.name() == "ethos_u_inference_advisor"
