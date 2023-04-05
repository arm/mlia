# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Argo backend module."""
import logging

from mlia.backend.argo.install import get_argo_installation
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory
from mlia.utils.misc import is_docker_available_cached

logger = logging.getLogger(__name__)

if is_docker_available_cached():
    # Argo requires docker. Only register the backend if docker is available,
    # e.g. to avoid docker-in-docker scenarios.
    registry.register(
        "argo",
        BackendConfiguration(
            supported_advice=[AdviceCategory.PERFORMANCE],
            supported_systems=[System.LINUX_AMD64],
            backend_type=BackendType.DOCKER,
            installation=get_argo_installation(),
        ),
        pretty_name="Argo",
    )
else:
    logger.warning(
        "Docker seems to be unavailable. The Argo backend cannot be registered."
    )
