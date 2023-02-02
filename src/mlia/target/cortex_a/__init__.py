# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A target module."""
from mlia.target.cortex_a.advisor import configure_and_get_cortexa_advisor
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo

registry.register(
    "cortex-a",
    TargetInfo(
        supported_backends=["ArmNNTFLiteDelegate"],
        default_backends=["ArmNNTFLiteDelegate"],
        advisor_factory_func=configure_and_get_cortexa_advisor,
        target_profile_cls=CortexAConfiguration,
    ),
)
