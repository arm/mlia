# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Argo backend module."""
import logging

from mlia.backend.argo.install import ARGO_PATH_ENV_VAR
from mlia.backend.argo.install import get_argo_installation
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

logger = logging.getLogger(__name__)

argo_installation = get_argo_installation()
if argo_installation.could_be_installed or argo_installation.executable_overwrite:
    # Argo requires docker or an env var pointing to a local Argo path. Only
    # register the backend if docker is available or if the env var is set to
    # avoid docker-in-docker scenarios.
    registry.register(
        "argo",
        BackendConfiguration(
            supported_advice=[AdviceCategory.PERFORMANCE],
            supported_systems=[System.LINUX_AMD64],
            backend_type=BackendType.DOCKER,
            installation=argo_installation,
        ),
        pretty_name="Argo",
    )
else:
    logger.warning(
        "The Argo backend cannot be registered. Docker seems to be unavailable "
        "or no local path is specified in environment variable %s.",
        ARGO_PATH_ENV_VAR,
    )
