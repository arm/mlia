# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Hydra target module."""
from mlia.target.hydra.advisor import configure_and_get_hydra_advisor
from mlia.target.hydra.config import HydraConfiguration
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo


registry.register(
    "hydra",
    TargetInfo(
        supported_backends=["argo"],
        default_backends=["argo"],
        advisor_factory_func=configure_and_get_hydra_advisor,
        target_profile_cls=HydraConfiguration,
    ),
)
