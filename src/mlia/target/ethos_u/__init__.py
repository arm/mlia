# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U target module."""
from mlia.backend.corstone import CORSTONE_PRIORITY
from mlia.target.ethos_u.advisor import configure_and_get_ethosu_advisor
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.config import get_default_ethos_u_backends
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo

SUPPORTED_BACKENDS_PRIORITY = ["Vela", *CORSTONE_PRIORITY]


for ethos_u in ("ethos-u55", "ethos-u65"):
    registry.register(
        ethos_u,
        TargetInfo(
            supported_backends=SUPPORTED_BACKENDS_PRIORITY,
            default_backends=get_default_ethos_u_backends(SUPPORTED_BACKENDS_PRIORITY),
            advisor_factory_func=configure_and_get_ethosu_advisor,
            target_profile_cls=EthosUConfiguration,
        ),
    )
