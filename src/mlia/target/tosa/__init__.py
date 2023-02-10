# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA target module."""
from mlia.target.registry import registry
from mlia.target.registry import TargetInfo
from mlia.target.tosa.advisor import configure_and_get_tosa_advisor
from mlia.target.tosa.config import TOSAConfiguration

registry.register(
    "tosa",
    TargetInfo(
        supported_backends=["tosa-checker"],
        default_backends=["tosa-checker"],
        advisor_factory_func=configure_and_get_tosa_advisor,
        target_profile_cls=TOSAConfiguration,
    ),
    pretty_name="TOSA",
)
